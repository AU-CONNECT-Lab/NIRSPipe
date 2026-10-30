"""``--bids-filter-file`` is read as UTF-8, whatever the machine's locale."""

from __future__ import annotations

import json

from nirspipe.io.bids import get_nirs_files


class _Layout:
    def get(self, **kwargs):
        self.kwargs = kwargs
        return []


def test_a_filter_file_saved_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "filter.json"
    path.write_text(json.dumps({"run": "01", "acquisition": "Région"}, ensure_ascii=False),
                    encoding="utf-8-sig")
    layout = _Layout()
    get_nirs_files(layout, "01", filter_file=path)
    assert layout.kwargs["run"] == "01"
    assert layout.kwargs["acquisition"] == "Région"
