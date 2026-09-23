# Walk: Inventio v1
Status: done
Updated: 2026-09-23

## Contract
- Outcome: một CLI local (`inventio`) cho Zero và agent: nạp source thành bản đồ trong một file SQLite ở cache, hỏi thì trả đoạn gốc kèm toạ độ. Nguồn: `docs/maps/laya-rag-cli/map.md` (Direction acceptance).
- Observable done: trên bộ markdown (40 câu), mỗi tầng và mỗi bộ chấm được đo so với BM25 trơn; tầng nào không thắng thì tắt; mỗi câu vài giây. Bộ thật (repo + Confluence) đo lại khi Zero đưa. Nguồn: map dòng 21.
- Scope: nguồn thư mục local (markdown, Python, text/code); cây + link do code; bộ chấm `none | laya | typesafe`; xuất nhãn Jev. Non-goals v1: Confluence connector, tầng category do model sinh, trục loại tri thức (D6 = c), fine-tune Laya.
- Boundaries: local là luật cứng; text source private không rời máy; file map không nằm trong repo nào.
- Verification: `inventio bench` trên `evidence/bench-md.jsonl`; test hợp đồng trong repo; smoke CLI thật.
- Open forks: không.

## Regions
| Region | Depends on | Visible outcome | Fog | Owner | Proof |
|---|---|---|---|---|---|
| Store + nạp + BM25 | — | `init`, `query` trả toạ độ đúng dòng | clear | Sophia | test toạ độ; bench `none` |
| Link (citation, mentions) | nạp | link hiện trong kết quả, nối được qua source | clear | Sophia | test prose → code, test anchor |
| Bộ chấm + guardrail | BM25 | `--ranker laya/typesafe`; typesafe từ chối source private | clear | Sophia | bench; test từ chối; smoke exit 3 |
| Đo so với done | tất cả | bảng số trong README | clear | Sophia | `evidence/bench-*.json` |
| Confluence, category, fine-tune | quyết định sau | — | deferred | — | — |

## Position
- Observed done: repo `C:/Users/LEGION/inventio` (push lên `github.com/minhquan23102000/inventio`). 5 test pass. Bench trên index cuối (126 file, 2.132 đoạn):
  - BM25 chỉ text: top-1 23, top-5 30, top-10 32.
  - BM25 + đường dẫn/heading (mặc định): 24 / 32 / 35. Tầng cây thắng → giữ.
  - Laya multilingual zero-shot: 12 / 28 / 30; 0,6 s/câu.
  - TypeSafe Jev: 32 / 36 / 37; 1,1 s/câu.
  - TypeSafe + mở rộng theo link: 30 / 36 / 37. Không kéo thêm đáp án nào vào pool (38/40 cả hai) → tắt mặc định, cờ `--links`.
- Smoke: `query --ranker typesafe` trả đúng đoạn p=0.96; `--ranker laya` chạy GPU; khi `docs` không `--public`, typesafe thoát mã 3, không gửi gì. `labels` xuất 2.373 nhãn Jev.
- Open assumptions: link `mentions` chỉ có giá trị khi code và văn xuôi dùng chung định danh — bộ markdown không có code nên chưa thử được đúng chỗ.
- Benchmark công khai (2026-09-23, chi tiết và cách chạy lại ở `benchmarks/README.md` trong repo), nDCG@10:
  - SciFact: BM25 0,670 (bài báo công bố 0,665, tức đường đo không tự tâng), Laya 0,302, TypeSafe 0,765.
  - StackOverflow QA (CoIR, văn bản trộn code): BM25 0,670 trên 1.994 câu (công bố 0,568); 300 câu đầu: BM25 0,713, Laya 0,203, TypeSafe 0,837.
  - SWE-bench Lite `code` (CodeRAG-Bench): BM25 0,540 (công bố 0,430), Laya 0,391, TypeSafe 0,696; có file cần sửa ở hạng 1 là 38% → 56%, trong top-5 là 64% → 78%.
  - SWE-bench Lite `mixed` (cả repo: code, test, docs): BM25 0,400, TypeSafe 0,515; trần pool tụt 0,823 → 0,633. Trên repo hỗn hợp, cái giới hạn là pool 30 ứng viên, không phải bộ chấm.
- Next move: Zero đưa repo + Confluence và 20-30 câu hỏi thật; đo lại `--links` trên bộ đó. Việc kế tiếp về kỹ thuật: pool cho repo hỗn hợp; tree-sitter cho SQL/Java/Scala (dbt, Flink), Python giữ `ast`.

## Decisions
- Link expansion mặc định tắt: đo trên markdown không thắng BM25 (correction trigger đầu tiên của map đã kích hoạt). Đảo lại nếu bộ repo + Confluence cho thấy nó kéo thêm đáp án vào pool.
- Bộ chấm mặc định `none`: Laya zero-shot tệ hơn BM25; TypeSafe là cloud nên phải bật tường minh.
- Tên: Inventio, lệnh `inventio` (2026-09-23). Zero bác `laya-atlas` ("tôi không thích", muốn "có tính sử thi"), rồi bác Lạc Thư ("nên đặt tiếng anh hoặc latin"). Inventio là canon thứ nhất của tu từ học, từ *invenire*, "tìm thấy": tìm chất liệu có sẵn trong các *loci*, khớp với nguyên tắc cấu trúc lấy từ chính source. Tên còn trống trên PyPI.
- Map ở `%LOCALAPPDATA%\inventio\map.db`, ngoài mọi repo.

## Evidence
- Bộ câu hỏi: `evidence/bench-md.jsonl` (40 câu, sinh bằng model nhỏ từ đoạn ngẫu nhiên ở 40 file khác nhau, seed 11).
- Kết quả: `evidence/bench-bm25-plain.json`, `bench-none.json`, `bench-none_links.json`, `bench-laya.json`, `bench-typesafe.json`, `bench-typesafe_links.json`.
- Test: `C:/Users/LEGION/inventio/tests/test_inventio.py` — toạ độ markdown/Python, link prose → code qua source, link tới heading, typesafe từ chối source private.

## Corrections
- Chunk từng tính dòng trống cuối file vào toạ độ (test anchor bắt được); sửa bằng cắt dòng trống hai đầu mỗi đoạn.
- `rebuild_links` mất 38 s trên astropy: index `idents(ident)` bắt join `mentions` quét mọi lần nhắc tới một tên phổ biến (bậc hai). Đổi sang index `(ident, role)`: 0,47 s, cùng 90.964 link.
- TypeSafe trả 403 "Attention Required" (tường lửa Cloudflare) cho một số đoạn code Django, và một đoạn bị chặn làm hỏng cả câu hỏi. Nay đoạn bị chặn không được chấm và giữ chỗ BM25 sau các đoạn đã chấm; lỗi 403 thật (key) vẫn raise.
