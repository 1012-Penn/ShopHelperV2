from pathlib import Path
import pytest
from app.services.quality.sources import SourceDocuments, source_url


def test_source_heading_and_traversal(tmp_path):
    (tmp_path/'policy.md').write_text('# 政策\n\n## 退款\n不能承诺到账时间。\n<script>alert(1)</script>')
    docs=SourceDocuments(tmp_path,{'policy.md'})
    url=source_url('doc:policy.md:hash',['政策','退款'])
    html=docs.render('policy.md')
    assert url.split('#')[1] in html
    assert '不能承诺到账时间' in html and '<script>' not in html
    with pytest.raises(FileNotFoundError):docs.render('../.env')
    with pytest.raises(FileNotFoundError):docs.render('unregistered.md')
