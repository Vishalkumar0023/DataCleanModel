from pathlib import Path

from upload_claim import claim_temp_upload


def test_only_one_request_can_claim_an_uploaded_file(tmp_path):
    temp_upload = tmp_path / "temp_upload.csv"
    temp_upload.write_text("name,value\nAda,1\n", encoding="utf-8")

    first_claim = claim_temp_upload(str(temp_upload))
    second_claim = claim_temp_upload(str(temp_upload))

    assert first_claim is not None
    assert second_claim is None
    assert not temp_upload.exists()
    assert Path(first_claim).read_text(encoding="utf-8") == "name,value\nAda,1\n"
