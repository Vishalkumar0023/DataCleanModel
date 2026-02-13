"""
Data Pipeline Web Application with JWT Authentication
======================================================
Flask web app with JWT token-based authentication and per-user dataset storage.
"""

import os
import json
import io
import base64
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, g, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import jwt as pyjwt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import DataPipeline, ModelTrainer, DataCleaner

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pipeline_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024  # 1GB

# ── JWT Configuration ──
JWT_SECRET_KEY = app.config['SECRET_KEY']
JWT_ALGORITHM = 'HS256'
JWT_ACCESS_EXPIRY = timedelta(minutes=30)
JWT_REFRESH_EXPIRY = timedelta(days=7)

# Create base folders
BASE_UPLOAD_FOLDER = 'user_data'
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)


# ==============================================================================
# DATABASE MODELS
# ==============================================================================

class User(db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with datasets
    datasets = db.relationship('Dataset', backref='owner', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @property
    def is_authenticated(self):
        return True


class Dataset(db.Model):
    """Model to store user datasets."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    cleaned_path = db.Column(db.String(500))
    final_path = db.Column(db.String(500))
    original_rows = db.Column(db.Integer)
    original_cols = db.Column(db.Integer)
    cleaned_rows = db.Column(db.Integer)
    cleaned_cols = db.Column(db.Integer)
    final_rows = db.Column(db.Integer)
    final_cols = db.Column(db.Integer)
    target_column = db.Column(db.String(100))
    problem_type = db.Column(db.String(50))
    processing_log = db.Column(db.Text)
    model_path = db.Column(db.String(500))
    model_results = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# ==============================================================================
# JWT HELPER FUNCTIONS
# ==============================================================================

def create_access_token(user_id):
    """Create a short-lived JWT access token."""
    payload = {
        'sub': str(user_id),
        'type': 'access',
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + JWT_ACCESS_EXPIRY
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id):
    """Create a long-lived JWT refresh token."""
    payload = {
        'sub': str(user_id),
        'type': 'refresh',
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + JWT_REFRESH_EXPIRY
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token):
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except pyjwt.ExpiredSignatureError:
        return None
    except pyjwt.InvalidTokenError:
        return None


def get_current_user():
    """
    Extract the current user from:
      1. HttpOnly cookie 'access_token'  (browser flow)
      2. Authorization: Bearer <token>   (API flow)
    Returns User object or None.
    """
    token = None
    
    # Try cookie first (browser flow)
    token = request.cookies.get('access_token')
    
    # Try Authorization header (API flow)
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
    
    if not token:
        return None
    
    payload = decode_token(token)
    if not payload or payload.get('type') != 'access':
        return None
    
    user = User.query.get(int(payload['sub']))
    return user


def jwt_required(f):
    """Decorator to protect routes – replaces @login_required."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            # For API requests return 401
            if request.is_json or request.headers.get('Authorization'):
                return jsonify({'error': 'Authentication required', 'code': 'TOKEN_EXPIRED'}), 401
            # For browser requests redirect to login
            return redirect(url_for('login'))
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def set_auth_cookies(response, user_id):
    """Set access and refresh token cookies on a response."""
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    
    response.set_cookie(
        'access_token', access_token,
        httponly=True, samesite='Lax',
        max_age=int(JWT_ACCESS_EXPIRY.total_seconds()),
        path='/'
    )
    response.set_cookie(
        'refresh_token', refresh_token,
        httponly=True, samesite='Lax',
        max_age=int(JWT_REFRESH_EXPIRY.total_seconds()),
        path='/'
    )
    return response


def clear_auth_cookies(response):
    """Clear authentication cookies."""
    response.delete_cookie('access_token', path='/')
    response.delete_cookie('refresh_token', path='/')
    return response


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_user_folder(user_id):
    """Get or create user-specific folder."""
    folder = os.path.join(BASE_UPLOAD_FOLDER, str(user_id))
    os.makedirs(folder, exist_ok=True)
    return folder


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def generate_plots(df, target_col=None):
    """Generate EDA plots and return as base64 images."""
    plots = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=len(numeric_cols) <= 10, fmt='.2f',
                       cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()
            plots['correlation'] = fig_to_base64(fig)
            plt.close(fig)
        except:
            pass
    
    # Distribution plots
    if numeric_cols:
        try:
            cols_to_plot = numeric_cols[:6]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
            axes = np.atleast_2d(axes).flatten()
            
            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                data = df[col].dropna()
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(data.mean(), color='red', linestyle='--', label='Mean')
                ax.axvline(data.median(), color='green', linestyle='--', label='Median')
                ax.set_title(col)
                ax.legend(fontsize=8)
            
            for idx in range(len(cols_to_plot), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Feature Distributions', fontsize=14)
            plt.tight_layout()
            plots['distributions'] = fig_to_base64(fig)
            plt.close(fig)
        except:
            pass
    
    return plots


# ==============================================================================
# AUTH ROUTES
# ==============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if request.is_json:
                resp = jsonify({
                    'success': True,
                    'redirect': url_for('dashboard'),
                    'access_token': create_access_token(user.id),
                    'refresh_token': create_refresh_token(user.id)
                })
            else:
                resp = make_response(redirect(url_for('dashboard')))
            
            set_auth_cookies(resp, user.id)
            return resp
        
        if request.is_json:
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('auth.html', mode='login')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page."""
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validate
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({'error': 'Username already exists'}), 400
            return render_template('auth.html', mode='signup')
        
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({'error': 'Email already registered'}), 400
            return render_template('auth.html', mode='signup')
        
        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Create user folder
        get_user_folder(user.id)
        
        if request.is_json:
            resp = jsonify({
                'success': True,
                'redirect': url_for('dashboard'),
                'access_token': create_access_token(user.id),
                'refresh_token': create_refresh_token(user.id)
            })
        else:
            resp = make_response(redirect(url_for('dashboard')))
        
        set_auth_cookies(resp, user.id)
        return resp
    
    return render_template('auth.html', mode='signup')


@app.route('/logout')
def logout():
    """Logout user — clear JWT cookies."""
    resp = make_response(redirect(url_for('login')))
    clear_auth_cookies(resp)
    return resp


@app.route('/api/refresh', methods=['POST'])
def refresh_token():
    """
    Refresh the access token using the refresh token.
    Reads from cookie or JSON body.
    """
    token = request.cookies.get('refresh_token')
    
    if not token and request.is_json:
        token = request.get_json().get('refresh_token')
    
    if not token:
        return jsonify({'error': 'Refresh token required'}), 401
    
    payload = decode_token(token)
    if not payload or payload.get('type') != 'refresh':
        return jsonify({'error': 'Invalid or expired refresh token', 'code': 'REFRESH_EXPIRED'}), 401
    
    user = User.query.get(int(payload['sub']))
    if not user:
        return jsonify({'error': 'User not found'}), 401
    
    new_access = create_access_token(user.id)
    
    resp = jsonify({
        'success': True,
        'access_token': new_access
    })
    resp.set_cookie(
        'access_token', new_access,
        httponly=True, samesite='Lax',
        max_age=int(JWT_ACCESS_EXPIRY.total_seconds()),
        path='/'
    )
    return resp


# ==============================================================================
# MAIN ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Home page - landing page for visitors, dashboard for logged-in users."""
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/dashboard')
@jwt_required
def dashboard():
    """User dashboard with their datasets."""
    datasets = Dataset.query.filter_by(user_id=g.current_user.id).order_by(Dataset.created_at.desc()).all()
    return render_template('dashboard.html', datasets=datasets, user=g.current_user)


@app.route('/upload', methods=['POST'])
@jwt_required
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        return jsonify({'error': 'Only CSV and Excel files are supported'}), 400
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Save to user folder
        user_folder = get_user_folder(g.current_user.id)
        temp_path = os.path.join(user_folder, 'temp_upload.csv')
        df.to_csv(temp_path, index=False)
        
        # Get column info
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        missing = {col: int(df[col].isnull().sum()) for col in columns}
        
        # Get sample data - handle NaN
        sample_df = df.head(5).replace({np.nan: None})
        sample = sample_df.to_dict('records')
        for row in sample:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif hasattr(value, 'item'):
                    row[key] = value.item()
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'shape': list(df.shape),
            'columns': columns,
            'dtypes': dtypes,
            'missing': missing,
            'sample': sample
        })
    
    except Exception as e:
        import traceback
        print(f"Upload error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
@jwt_required
def process_data():
    """Process uploaded data through the pipeline."""
    try:
        data = request.json
        target_col = data.get('target_column')
        problem_type = data.get('problem_type', 'regression')
        dataset_name = data.get('dataset_name', 'Untitled Dataset')
        original_filename = data.get('original_filename', 'unknown.csv')
        
        # Load from user's temp file
        user_folder = get_user_folder(g.current_user.id)
        temp_path = os.path.join(user_folder, 'temp_upload.csv')
        
        if not os.path.exists(temp_path):
            return jsonify({'error': 'No file uploaded. Please upload a file first.'}), 400
        
        # Run pipeline
        pipeline = DataPipeline()
        pipeline.load(temp_path)
        
        # Validate
        validation = pipeline.validate()
        
        # Clean
        pipeline.clean()
        cleaning_summary = pipeline.cleaner.get_cleaning_summary()
        
        # Feature engineering
        if target_col and target_col in pipeline.cleaned_df.columns:
            pipeline.engineer_features(target_col=target_col, problem_type=problem_type)
            feature_summary = pipeline.engineer.get_summary()
        else:
            pipeline.engineer_features()
            feature_summary = pipeline.engineer.get_summary() if pipeline.engineer else {}
        
        # Save results with unique names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cleaned_filename = f'cleaned_{timestamp}.csv'
        final_filename = f'final_{timestamp}.csv'
        
        cleaned_path = os.path.join(user_folder, cleaned_filename)
        final_path = os.path.join(user_folder, final_filename)
        
        pipeline.cleaned_df.to_csv(cleaned_path, index=False)
        pipeline.final_df.to_csv(final_path, index=False)
        
        # Save to database
        dataset = Dataset(
            name=dataset_name,
            original_filename=original_filename,
            cleaned_path=cleaned_path,
            final_path=final_path,
            original_rows=validation['shape'][0],
            original_cols=validation['shape'][1],
            cleaned_rows=pipeline.cleaned_df.shape[0],
            cleaned_cols=pipeline.cleaned_df.shape[1],
            final_rows=pipeline.final_df.shape[0],
            final_cols=pipeline.final_df.shape[1],
            target_column=target_col,
            problem_type=problem_type,
            processing_log=json.dumps({
                'cleaning': cleaning_summary['operations'],
                'feature_engineering': feature_summary.get('transformations', [])
            }),
            user_id=g.current_user.id
        )
        db.session.add(dataset)
        db.session.commit()
        
        # Generate plots
        plots = generate_plots(pipeline.cleaned_df, target_col)
        
        # Remove temp file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'dataset_id': dataset.id,
            'validation': {
                'original_shape': validation['shape'],
                'missing_count': validation['missing_values']['total_missing_cells'],
                'duplicate_count': validation['duplicates']['count']
            },
            'cleaning': {
                'final_shape': list(pipeline.cleaned_df.shape),
                'operations': cleaning_summary['operations']
            },
            'feature_engineering': {
                'final_shape': list(pipeline.final_df.shape),
                'transformations': feature_summary.get('transformations', [])
            },
            'plots': plots
        })
    
    except Exception as e:
        import traceback
        print(f"Process error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/<int:dataset_id>')
@jwt_required
def view_dataset(dataset_id):
    """View a specific dataset."""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Ensure user owns this dataset
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    # Load samples for preview
    cleaned_sample = []
    cleaned_columns = []
    final_sample = []
    final_columns = []
    
    try:
        if os.path.exists(dataset.cleaned_path):
            df_clean = pd.read_csv(dataset.cleaned_path)
            cleaned_columns = df_clean.columns.tolist()
            # Replace NaN with None for Jinja
            cleaned_sample = df_clean.head(10).replace({np.nan: None}).to_dict('records')
            
        if os.path.exists(dataset.final_path):
            df_final = pd.read_csv(dataset.final_path)
            final_columns = df_final.columns.tolist()
            final_sample = df_final.head(10).replace({np.nan: None}).to_dict('records')
    except Exception as e:
        print(f"Error loading dataset samples: {e}")
        
    return render_template('view_dataset.html', 
        dataset=dataset,
        cleaned_columns=cleaned_columns,
        cleaned_sample=cleaned_sample,
        final_columns=final_columns,
        final_sample=final_sample,
        processing_log=json.loads(dataset.processing_log) if dataset.processing_log else {}
    )


@app.route('/api/dataset/<int:dataset_id>/transform', methods=['POST'])
@jwt_required
def transform_dataset(dataset_id):
    """Apply data transformations."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    data = request.json
    operation = data.get('operation')
    params = data.get('params', {})
    
    # Load pipeline (just to use loader/cleaner logic)
    pipeline = DataPipeline()
    # Use cleaned path as starting point
    if os.path.exists(dataset.cleaned_path):
        pipeline.load(dataset.cleaned_path)
    else:
        return jsonify({'error': 'Dataset file not found'}), 404
        
    # Initialize cleaner with current data
    cleaner = DataCleaner(pipeline.raw_df)
    
    try:
        if operation == 'clean_numeric_text':
            cleaner.clean_numeric_text(**params)
        elif operation == 'rename_columns':
            cleaner.rename_columns(**params)
        elif operation == 'extract_regex':
            cleaner.extract_regex_feature(**params)
        elif operation == 'remove_duplicates':
            cleaner.remove_duplicates(**params)
        elif operation == 'drop_columns':
            cleaner.drop_columns(**params)
        else:
            return jsonify({'error': 'Invalid operation'}), 400
            
        # Get updated dataframe
        new_df = cleaner.get_cleaned_data()
        
        # Save updated file (overwriting cleaned path for now)
        cleaned_path = dataset.cleaned_path
        new_df.to_csv(cleaned_path, index=False)
        
        # Also update final path if we are treating them similarly, 
        # or just let future feature engineering handle it. 
        # For this demo, let's keep them in sync if no detailed FE has been done yet.
        if os.path.exists(dataset.final_path):
            new_df.to_csv(dataset.final_path, index=False)
            dataset.final_rows = new_df.shape[0]
            dataset.final_cols = new_df.shape[1]
        
        # Update metadata
        dataset.cleaned_rows = new_df.shape[0]
        dataset.cleaned_cols = new_df.shape[1]
        
        # Add to log
        try:
            log = json.loads(dataset.processing_log) if dataset.processing_log else {}
        except:
            log = {}
            
        if 'cleaning' not in log: log['cleaning'] = []
        
        # Get last operation from cleaner log
        if cleaner.cleaning_log:
            log['cleaning'].extend(cleaner.cleaning_log[-1:])
            
        dataset.processing_log = json.dumps(log)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'preview': new_df.head().replace({np.nan: None}).to_dict(orient='records'),
            'columns': new_df.columns.tolist(),
            'stats': {
                'rows': new_df.shape[0],
                'cols': new_df.shape[1]
            },
            'message': cleaner.cleaning_log[-1] if cleaner.cleaning_log else "Transformation applied"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    # Load cleaned data preview
    cleaned_df = pd.read_csv(dataset.cleaned_path)
    final_df = pd.read_csv(dataset.final_path)
    
    # Get sample data
    cleaned_sample = cleaned_df.head(10).replace({np.nan: None}).to_dict('records')
    final_sample = final_df.head(10).replace({np.nan: None}).to_dict('records')
    
    # Parse processing log
    processing_log = json.loads(dataset.processing_log) if dataset.processing_log else {}
    
    return render_template('view_dataset.html', 
                          dataset=dataset,
                          cleaned_columns=cleaned_df.columns.tolist(),
                          final_columns=final_df.columns.tolist(),
                          cleaned_sample=cleaned_sample,
                          final_sample=final_sample,
                          processing_log=processing_log)


@app.route('/dataset/<int:dataset_id>/download/<file_type>')
@jwt_required  
def download_dataset(dataset_id, file_type):
    """Download dataset file."""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Check for format query parameter (csv or xlsx)
    file_format = request.args.get('format', 'csv')
    
    if file_type == 'cleaned':
        path = dataset.cleaned_path
        base_filename = f'{dataset.name}_cleaned'
    elif file_type == 'final':
        path = dataset.final_path
        base_filename = f'{dataset.name}_model_ready'
    elif file_type == 'model':
        path = dataset.model_path
        # Models are always .pkl downloads
        filename = f'{dataset.name}_model.pkl'
        if not path or not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(path, as_attachment=True, download_name=filename)
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    # Handle CSV download (default)
    if file_format == 'csv':
        filename = f'{base_filename}.csv'
        return send_file(path, as_attachment=True, download_name=filename)
        
    # Handle Excel download
    elif file_format == 'xlsx':
        try:
            # Read CSV and convert to Excel in memory
            df = pd.read_csv(path)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            
            filename = f'{base_filename}.xlsx'
            return send_file(
                output,
                as_attachment=True,
                download_name=filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            return jsonify({'error': f'Error converting to Excel: {str(e)}'}), 500
            
    else:
        return jsonify({'error': 'Invalid format requested'}), 400


@app.route('/dataset/<int:dataset_id>/delete', methods=['POST'])
@jwt_required
def delete_dataset(dataset_id):
    """Delete a dataset."""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Delete files
    try:
        if dataset.cleaned_path and os.path.exists(dataset.cleaned_path):
            os.remove(dataset.cleaned_path)
        if dataset.final_path and os.path.exists(dataset.final_path):
            os.remove(dataset.final_path)
        if dataset.model_path and os.path.exists(dataset.model_path):
            os.remove(dataset.model_path)
    except:
        pass
    
    # Delete from database
    db.session.delete(dataset)
    db.session.commit()
    
    return jsonify({'success': True})


@app.route('/api/datasets')
@jwt_required
def api_datasets():
    """API endpoint to get user's datasets."""
    datasets = Dataset.query.filter_by(user_id=g.current_user.id).order_by(Dataset.created_at.desc()).all()
    
    return jsonify([{
        'id': d.id,
        'name': d.name,
        'original_filename': d.original_filename,
        'original_rows': d.original_rows,
        'original_cols': d.original_cols,
        'final_rows': d.final_rows,
        'final_cols': d.final_cols,
        'target_column': d.target_column,
        'problem_type': d.problem_type,
        'created_at': d.created_at.isoformat()
    } for d in datasets])


# ==============================================================================
# MODEL TRAINING ROUTE
# ==============================================================================

@app.route('/dataset/<int:dataset_id>/train', methods=['POST'])
@jwt_required
def train_model(dataset_id):
    """Train ML models on a dataset."""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Load final (or cleaned) data
        data_path = dataset.final_path or dataset.cleaned_path
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Dataset file not found'}), 404
        
        df = pd.read_csv(data_path)
        
        # Get parameters
        data = request.json or {}
        target_col = data.get('target_column', dataset.target_column)
        problem_type = data.get('problem_type', dataset.problem_type)
        
        # Run model trainer
        user_folder = get_user_folder(g.current_user.id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{dataset_id}_{timestamp}.pkl'
        model_path = os.path.join(user_folder, model_filename)
        
        trainer = ModelTrainer(df, target_col=target_col, problem_type=problem_type)
        results = trainer.run()
        trainer.export_model(model_path)
        
        # Save results to database
        dataset.model_path = model_path
        dataset.model_results = json.dumps(results, default=str)
        if trainer.target_col:
            dataset.target_column = trainer.target_col
        if trainer.problem_type:
            dataset.problem_type = trainer.problem_type
        db.session.commit()
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        import traceback
        print(f"Training error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# FILE MANAGEMENT ROUTES
# ==============================================================================

@app.route('/files')
@jwt_required
def file_manager():
    """Render the file manager page."""
    return render_template('files.html')

@app.route('/api/user_files')
@jwt_required
def get_user_files():
    """Get list of files in user's directory."""
    user_folder = get_user_folder(g.current_user.id)
    files = []
    
    if os.path.exists(user_folder):
        for entry in os.scandir(user_folder):
            if entry.is_file() and not entry.name.startswith('.'):
                try:
                    stat = entry.stat()
                    file_type = 'Unknown'
                    if entry.name.endswith('.pkl'):
                        file_type = 'Model (.pkl)'
                    elif entry.name.endswith('.csv'):
                        if 'cleaned' in entry.name:
                            file_type = 'Cleaned Data (.csv)'
                        elif 'final' in entry.name:
                            file_type = 'Model-Ready Data (.csv)'
                        else:
                            file_type = 'Raw Data (.csv)'
                    
                    files.append({
                        'name': entry.name,
                        'size': stat.st_size,
                        'date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_type
                    })
                except Exception as e:
                    print(f"Error reading file {entry.name}: {e}")

    # Sort by date desc
    files.sort(key=lambda x: x['date'], reverse=True)
    return jsonify(files)

@app.route('/api/delete_files', methods=['POST'])
@jwt_required
def delete_user_files():
    """Bulk delete files."""
    data = request.json
    filenames = data.get('filenames', [])
    user_folder = get_user_folder(g.current_user.id)
    deleted = []
    errors = []
    
    for filename in filenames:
        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            errors.append(f"Invalid filename: {filename}")
            continue
            
        path = os.path.join(user_folder, filename)
        try:
            if os.path.exists(path):
                os.remove(path)
                deleted.append(filename)
            else:
                errors.append(f"File not found: {filename}")
        except Exception as e:
            errors.append(f"Error deleting {filename}: {str(e)}")
            
    return jsonify({'deleted': deleted, 'errors': errors})


# ==============================================================================
# INIT DATABASE
# ==============================================================================

with app.app_context():
    db.create_all()


# ==============================================================================
# DEMO & PREDICTION ROUTE
# ==============================================================================

DEMO_MODELS_DIR = 'demo_models'

def load_demo_model(model_type):
    """Load a demo model (cached if possible)."""
    try:
        if model_type not in ['sales', 'student']:
            return None
        
        filename = f'{model_type}_model.pkl'
        path = os.path.join(DEMO_MODELS_DIR, filename)
        
        if not os.path.exists(path):
            return None
            
        import joblib
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading demo model: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict_demo():
    """
    Public API endpoint for demo predictions.
    Does NOT require JWT auth to allow easy testing.
    Payload: { "model_type": "sales"|"student", "features": {...} }
    """
    try:
        data = request.json
        model_type = data.get('model_type')
        features = data.get('features')
        
        if not model_type or not features:
            return jsonify({'error': 'Missing model_type or features'}), 400
            
        model = load_demo_model(model_type)
        if not model:
            return jsonify({'error': 'Model not found or could not be loaded'}), 404
            
        # Prepare input data
        # Expecting features to be a dict, convertible to DataFrame for sklearn
        # e.g. {"TV_Ad_Budget": 100, ...}
        
        # Ensure correct feature order/names based on training
        if model_type == 'sales':
            feature_names = ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
        elif model_type == 'student':
            feature_names = ['Study_Hours', 'Attendance_Percentage', 'Previous_Score']
        else:
            return jsonify({'error': 'Unknown model type'}), 400
            
        # Create DataFrame
        try:
            input_df = pd.DataFrame([features])
            # Select/Reorder columns
            input_df = input_df[feature_names]
        except KeyError as e:
            return jsonify({'error': f'Missing feature: {str(e)}'}), 400
            
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'prediction': float(prediction)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo_page():
    """Render the public demo page."""
    user = get_current_user()
    user_models = []
    untrained_datasets = []
    
    if user:
        # Fetch all user datasets
        datasets = Dataset.query.filter_by(user_id=user.id).all()
        for d in datasets:
            info = {
                'id': d.id,
                'name': d.name,
                'target': d.target_column,
                'type': d.problem_type,
                'created_at': d.created_at.strftime('%Y-%m-%d')
            }
            
            if d.model_path and os.path.exists(d.model_path):
                user_models.append(info)
            else:
                untrained_datasets.append(info)
                
    return render_template('demo.html', user=user, user_models=user_models, untrained_datasets=untrained_datasets)


@app.route('/api/user_model_info/<int:dataset_id>')
@jwt_required
def get_user_model_info(dataset_id):
    """Get metadata for a user's trained model."""
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({'error': 'Access denied'}), 403
        
    if not dataset.model_path or not os.path.exists(dataset.model_path):
        return jsonify({'error': 'Model not found'}), 404
        
    try:
        import joblib
        model_data = joblib.load(dataset.model_path)
        
        # Extract metadata
        # model_data is a dict with keys: scalar, label_encoder, feature_names, etc.
        return jsonify({
            'success': True,
            'name': dataset.name,
            'features': model_data.get('feature_names', []),
            'target': dataset.target_column,
            'problem_type': dataset.problem_type,
            'metrics': model_data.get('metrics', {})
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_user_model', methods=['POST'])
@jwt_required
def predict_user_model():
    """Predict using a user's trained model."""
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        features = data.get('features')
        
        if not dataset_id or not features:
            return jsonify({'error': 'Missing dataset_id or features'}), 400
            
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != g.current_user.id:
            return jsonify({'error': 'Access denied'}), 403
            
        if not dataset.model_path or not os.path.exists(dataset.model_path):
            return jsonify({'error': 'Model not found'}), 404
            
        import joblib
        model_data = joblib.load(dataset.model_path)
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        
        # Prepare input
        input_df = pd.DataFrame([features])
        
        # Ensure columns match training data
        # Fill missing with 0 or mean? 0 for now
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder
        input_df = input_df[feature_names]
        
        # Scale
        if scaler:
            input_df = scaler.transform(input_df)
            
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'target': dataset.target_column
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# RUN APP
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DATA PIPELINE WEB APP (JWT Authentication)")
    print("=" * 60)
    print("\n🌐 Open in browser: http://127.0.0.1:8080")
    print("🔐 Auth: JWT tokens in HttpOnly cookies\n")
    app.run(debug=True, host='127.0.0.1', port=8080)
