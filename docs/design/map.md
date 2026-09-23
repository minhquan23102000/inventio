# Map: CLI truy xuất tri thức cục bộ bằng Laya (tên chưa chốt)
Status: ready-for-planning
Updated: 2026-09-23

## Hợp đồng định hướng
- Mapping invoked because: Zero gọi `map-the-unknowns` và nói hướng còn mờ: "Ý tôi còn hơi mơ hồ Sophia, tôi chỉ định hình phương hướng thôi chưa rõ lắm."
- Vai trò artifact: bản ghi định hướng đang sống; giữ cái đã biết, cái đã đo, cái còn mở. Không phải giấy phép build.
- Boundary: không viết code CLI, không chia việc. Probe đo đạc được phép (Zero: "được dùng pip tạm vậy").
- Next move: giao cho `walk-the-map`; bản ghi đi đường ở `docs/walks/inventio/walk.md`.

### Trạng thái đích
- [user-confirmed] Object: một CLI dùng Laya, đóng gói để đẩy lên GitHub; lưu bản đồ/storage vào file local. Evidence: "i want to build a cli that use laya"; "đóng gói gì đó thành cli để tôi đẩy lên github"; "tôi hình dung đó là cli, embedded mấy cái file , map hoặc storage vào local file là được ?" (2026-09-23).
- [user-confirmed] Ràng buộc: không embedding; BM25 được phép; local, rẻ, nhanh. Evidence: "RAG without any embedding ... 100% local, 100% cheap and fasst"; "không embedding , BM25 chấp nhận".
- [user-confirmed] Người đọc kết quả: cả agent (pilot / calibrated-judgment) lẫn người. Evidence: "Q1 cả hai"; "one of use case we dicuss so far is pilot extionsion and @.agents/skills/calibrated-judgment/".
- [user-confirmed] Source đầu tiên: một repo và Confluence. Evidence: "Q3. một repo và confluence". Các source sau: Slack, mail. Hook tự nạp để sau: "nâng cao để sau cũng được".
- [user-confirmed] Dữ liệu hỗn hợp; repo GitHub có cả private lẫn public. Evidence: "dữ liệu hỗn hợp, github private có public cũng có luôn".
- [user-confirmed] Cơ chế Zero cảm thấy: tiền xử lý là trọng tâm; bản đồ có category dạng cây, có link, trước ranker. Evidence: "quan trọng nhất là ở bước tiền xử lý đúng không?"; "cấu trúc cateogry không chỉ list mà là theo cây? link với nhau? vậy nó phải là dạng graph rag đúng không ta?"
- [user-confirmed] Zero cân nhắc: category động do model local nhẹ (MiniCPM) sinh; TypeSafe tạm thời nếu Laya chậm. Evidence: "category chỉ có thể cứng? hay kêu coding agent tự sinh ra?"; "https://github.com/openbmb/minicpm được không model nào nhẹ càng tốt? vậy thì khi đó category có thể động được ?"; "BM25 + laya (hoặc dùng typesafe ai tạm thời nếu nó chậm? )".
- [user-confirmed] Thứ tự giá trị: local là luật cứng; đúng > nhanh > rẻ. Kèm ngoại lệ có chủ đích: trong giai đoạn thử và demo, bộ chấm có tuỳ chọn dùng TypeSafe; Laya fine-tune sau. Evidence: "1. đúng được nhưng để test tạm thời có option dùng type safe ai sau đó fine tune laya sau?, vì đi demo cho sếp mà dùng laya nó chạy ra tệ là không được?" (2026-09-23).
- [user-confirmed] Bộ đo tạm thời: markdown, Confluence đưa sau. Evidence: "Confluence chưa nữa, lát tôi đưa, giờ bạn tạm bạn dùng markdown thử trước?"
- [user-confirmed] Done: trên bộ câu hỏi thật (repo + Confluence), đáp án đúng nằm trong top-5; mỗi câu vài giây; mỗi tầng bản đồ và mỗi bộ chấm phải thắng BM25 trơn trên cùng bộ, tầng nào không thắng thì bỏ. Trước khi có Confluence, đo trên bộ markdown. Evidence: "được" (2026-09-23), trả lời mục 1 của playback cuối.

### Vị trí hiện tại (đã quan sát)

Laya, từ nguồn:
- [observed] Không sinh chữ; trả `choice | score | noul` trong một forward pass. Source: README `NandhaKishorM/laya`.
- [observed] `laya` (en): context 512, `head_max_len` 192. `laya-multilingual`: 1024 / 256. Source: `cfg` sau khi load.
- [observed] Tác giả: "a fast base to specialise, not a zero-shot decision engine"; fine-tune (Kaggle 2xT4, 4-5 giờ) mới ra giá trị. Source: README Honest limits, Fine-Tuning.
- [observed] `choice` rớt khi quá ~20 lựa chọn. Source: README (Banking77 0.425).
- [observed] `predict_shortlist` của Laya dùng embedding cosine. Source: `laya/shortlist.py`.

Máy của Zero:
- [observed] RTX 5070 Laptop 8 GB, CUDA chạy; Python 3.14; torch 2.11+cu128; `laya 0.3.6`. SQLite của Python có FTS5 và hàm `bm25()` sẵn. Source: `nvidia-smi`, `pip`, truy vấn thử FTS5.

MiniCPM, từ nguồn:
- [observed] Bản nhẹ hiện có: MiniCPM5-1B (1.08 tỷ tham số, đếm khi load), MiniCPM5-2B (2.5 tỷ, context 131k), MiniCPM4-0.5B; có GGUF. Apache-2.0. Source: README `OpenBMB/MiniCPM`.

Probe, kho 298 đoạn cắt từ ba file thật (`calibrated-judgment/SKILL.md`, `docs/maps/jev-harness/map.md`, `docs/maps/judgment-routing/map.md`). Giới hạn chung: kho nhỏ, câu hỏi và đáp án do agent chọn; là probe để định hướng, không phải benchmark.

Truy vấn (8 câu, 5 EN / 3 VI, `noul` "Does this passage contain the answer to the question: …?"):
- [observed] Laya chấm thẳng cả kho: kém. Hạng đáp án đúng, `laya-multilingual`: 3, 2, 6, 2, 8, 1, 108, 8; `laya` (en) sụp với tiếng Việt (66, 114, 31).
- [observed] Xác suất `noul` bão hoà: 13-109 đoạn trên 298 được chấm > 0.5 mỗi câu.
- [observed] BM25: đáp án đúng hạng 1 ở 6/8 câu (hai câu còn lại hạng 2 và 12).
- [observed] Laya xếp lại top-20 của BM25: câu diễn đạt khác chữ gốc từ hạng 12 lên 1; một câu khác từ hạng 1 xuống 12. Trộn 50/50: 1, 1, 1, 2, 1, 1, 10, 2.
- [observed] Câu hỏi → chọn 1 trong 3 tài liệu: `laya` 6/8, `laya-multilingual` 5/8, confidence thấp.
- [observed] Tốc độ `laya-multilingual` trên GPU: ~7.6 ms/đoạn theo lô 32; lô 128 treo máy 8 GB.

Phân loại lúc nạp (298 đoạn → "đoạn này thuộc tài liệu nào", 3 lựa chọn có mô tả, ngẫu nhiên = 33%; đáp án biết sẵn từ nguồn gốc file):
- [observed] `laya-multilingual`: 83/298 (28%), dồn gần hết vào một nhãn; trong 169 đoạn nó tự tin ≥ 0.5 chỉ 46 đúng. Tự tin mà sai.
- [observed] `laya` (en): 132/298 (44%), gần như không bao giờ tự tin.
- [observed] MiniCPM5-1B (bf16, transformers, greedy): 90/298 (30%), trả "skill" cho 295/298 đoạn. ~75 ms/đoạn.
- [observed] Jev (`typesafe/jev-1.13.0`): 156/298 (52%); 110/179 đúng khi tự tin ≥ 0.5. 11 s, $0.006 cho 298 đoạn.

Đề xuất cây category, và chỉ mục BM25 (2026-09-23, lượt 3):
- [observed] MiniCPM5-1B, đưa 71 heading của ba file, bảo gom thành cây theo chủ đề: 9.4 s, trả về chính các heading lồng thành bậc thang, không gom chủ đề nào. Không dùng được.
- [observed] MiniCPM5-2B: probe làm sập kernel lúc load (nguyên nhân chưa rõ; nghi hết bộ nhớ khi còn model khác trên GPU). Chưa có kết quả.
- [observed] Chỉ mục BM25 trong SQLite FTS5 trên 298 đoạn: 2.614 từ; mỗi dòng là (từ, số đoạn chứa, số lần xuất hiện), ví dụ `model` → 33 đoạn / 45 lần, `pixel` → 1 / 1. Truy vấn "hook model" trả ba dòng đúng chủ đề hook đứng đầu. Không có model, không có vector.
- [observed] `Minibase/NER-Standard`: model sinh 135M tham số (GGUF Q8, 369 MB), chỉ tiếng Anh, nhận bốn loại PERSON / ORG / LOC / MISC; số đo tự báo trên 100 mẫu bộ dữ liệu riêng (F1 0.951, recall 1.000). Source: model card.
- [observed] Category động không dùng model (đồ thị item–từ–item từ chính tần suất từ, chia cộng đồng Louvain, 316 đoạn, 669 từ đặc trưng): ra 4 cụm, **chia theo ngôn ngữ chứ không theo chủ đề** (cụm tiếng Việt 140, cụm tiếng Anh 122); nhãn là từ đệm ("chưa · người · phải", "about · read · cannot"). Chỉ một cụm ra chủ đề thật: "credential · `judge` · `omp` · docs · fresh · session". Bản thô này không dùng được.
- [inferred] Nếu gom cụm theo định danh (tên bảng, hàm, file, lệnh) thay vì theo từ thường, sẽ không bị tách theo ngôn ngữ, vì định danh giống nhau trong văn tiếng Việt và tiếng Anh. Confidence: low-medium. Reason: cụm duy nhất có nghĩa ở trên chính là cụm gom quanh định danh (`judge`, `omp`). Chưa đo.

Người khác làm RAG bằng Jev thế nào (nghiên cứu 2026-09-23):
- [observed] Công thức chính TypeSafe công bố: BM25 lọc 30 ứng viên → mỗi cặp (câu hỏi, ứng viên) một câu `Noul` có tiêu chí đúng/sai viết rõ ("The candidate states the specific rule the query cites" / "The candidate is only on a similar topic") → sắp theo noul. Không embedding. Trên CLERC (3.565 đoạn án lệ, 40 câu): top-1 từ 5% lên 18%, top-10 từ 38% lên 62%. Source: docs.typesafe.ai/cookbooks/rerank_typesafe.
- [observed] Spring AI TypeSafe: `JevDocumentFilter` (sàng lọc liên quan + prompt injection) chạy trước `JevDocumentReranker` (một lời gọi mỗi tài liệu). Rẻ vì không sinh chữ, nhiều câu hỏi một lần gọi, và chỉ chấm shortlist. Source: spring.io/blog/2026/09/21/spring-ai-typesafe-structured-judgment; spring-ai-community.github.io/spring-ai-typesafe/latest/rag/JevDocumentReranker.
- [observed] TrustGraph: ontology nhỏ (5-10 lớp) rút tốt hơn cây 50 lớp; nên dựa trên ontology chuẩn (Dublin Core cho metadata tài liệu, PROV-O cho nguồn gốc, FIBO cho ngân hàng); chuỗi nguồn gốc Document → Page → Chunk → Subgraph. Nhưng việc rút của họ dùng LLM và vector similarity. Source: trustgraph.ai/guides/key-concepts/ontologies-and-context-graphs.
- [inferred] Hướng của bản đồ này trùng công thức chuẩn của TypeSafe, chỉ thay Jev bằng Laya chạy local. Probe Laya đầu tiên chưa có tiêu chí đúng/sai trong câu `noul`; đó là cải tiến rẻ cần thử. Confidence: high về cấu trúc, chưa đo về tiêu chí.

Category loại tri thức cố định (probe 2026-09-23): 189 dòng lấy từ bốn map, nhãn gốc là thẻ `[observed]/[inferred]/[hypothesis]/[user-confirmed]` (bỏ thẻ đi) → fact / inference / hypothesis / decision; 4 lựa chọn có mô tả; ngẫu nhiên 25%, đoán nhãn nhiều nhất 48%. Giới hạn: thẻ do agent gắn, ranh giới fact / inference vốn mờ.
- [observed] `laya` 65/189 (34%); `laya-multilingual` 46/189 (24%, dồn vào hypothesis 115 lần); Jev 109/189 (58%), $0.004.
- [observed] Chi phí tìm link bằng Laya kiểu mọi cặp: 316 item ≈ 50 nghìn cặp ≈ 6 phút; 10 nghìn item ≈ 50 triệu cặp ≈ 4,4 ngày ở 7.6 ms/cặp (phép nhân từ tốc độ đo).
- [inferred] Laya không tìm được link (bình phương số item), nhưng có thể gán **loại** cho link mà code đề xuất (cặp chung định danh, chung URL, hàng xóm BM25): "triển khai / mô tả / mâu thuẫn / không liên quan". Confidence: medium. Chưa đo.
- [inferred] Category chung mặc định chỉ tồn tại cho hai trục, không cho trục chủ đề: (1) metadata kiểu Dublin Core (nguồn, tác giả, ngày, space, loại file) do code lấy miễn phí; (2) loại tri thức (fact / requirement / decision / procedure / definition…) cố định cho mọi kho. Trục chủ đề luôn riêng từng miền, lấy từ cấu trúc + định danh. Confidence: medium-high. Reason: phân loại theo nhiều trục (faceted) là thực hành chuẩn; TrustGraph khuyên tập nhỏ; probe cho thấy model zero-shot yếu cả ở trục loại.
- [inferred] Trục loại tri thức khớp thẳng vào state của calibrated-judgment: requirement → `standard`, fact → `evidence`, decision / supersession → `counter`. Confidence: medium. Reason: đối chiếu với SKILL.md dòng 143-152.

Bộ đo markdown (2026-09-23): 116 file markdown trong `~/.omp` (docs, skills, friction, README; bỏ chính map này), cắt theo heading và đoạn ≤ 1.500 ký tự → 2.104 đoạn. 40 câu hỏi do một LLM viết, mỗi câu từ một đoạn lấy ngẫu nhiên ở 40 file khác nhau, yêu cầu "câu một đồng nghiệp chưa đọc đoạn này sẽ gõ, dùng từ tự nhiên, không chép cụm dài"; đáp án đúng là đoạn gốc. BM25 lấy top-30; các bộ chấm xếp lại top-30 đó. Giới hạn: câu hỏi do model sinh từ chính đoạn đáp án, nên có thể dễ cho BM25 hơn câu người thật; n=40.
- [observed] BM25 (0.3 ms/câu): đáp án trong top-30 37/40; top-1 21; top-5 30; top-10 33.
- [observed] Laya multilingual, câu `noul` trơn: top-1 11; top-5 28 (0.63 s/câu, GPU). Có tiêu chí đúng/sai: top-1 14; top-5 29. Laya en có tiêu chí: 13 / 25. **Laya zero-shot làm thứ hạng tệ hơn BM25.**
- [observed] Jev, cùng câu hỏi có tiêu chí đúng/sai: top-1 **32/40**; top-5 **37/40** (bằng trần, mọi đáp án BM25 tìm được đều lên top-5); top-10 37. 1.200 cặp, 23 s cả mẻ, $0.033 (≈ $0.0008/câu hỏi).
- [observed] Trộn thứ hạng BM25 + bộ chấm (reciprocal rank fusion): với Jev top-1 28 (kém Jev đơn); với Laya top-1 19, top-5 31.
- [observed] Ba câu đáp án không lọt top-30 của BM25: đây là trần của tầng lọc thô; chỉ cây / link / cách diễn đạt khác mới kéo lên được.
- [inferred] Tuỳ chọn TypeSafe cho demo là đúng và đo được: nó biến 21/40 thành 32/40 ở top-1. Laya phải fine-tune mới vào được vai này; 1.200 cặp Jev vừa chấm là mẫu đầu tiên cho dữ liệu fine-tune (dữ liệu public). Confidence: high cho hướng, medium cho độ lớn (n=40, câu do model sinh).

schema.org làm category chuẩn (2026-09-23):
- [observed] 826 type, 1.540 property, xếp thành cây; sinh ra để trang web mô tả *vật* mình nói tới (Product, Event, Recipe, Person…). Source: schema.org/docs/schemas.html.
- [observed] Nhánh `CreativeWork` có sẵn thuộc tính dùng được cho bản đồ: `about` (chủ đề chính), `mentions` (có nhắc tới nhưng không phải chủ đề chính), `isPartOf` / `hasPart` (cây), `citation`, `isBasedOn` (link), `author`, `dateModified`, `inLanguage`, `keywords`, `creativeWorkStatus` (Draft / Published / Obsolete). Source: schema.org/CreativeWork.
- [inferred] schema.org hợp làm **từ vựng cho link và metadata**, không hợp làm cây category: 826 type vượt xa giới hạn ~20 lựa chọn của Laya và lời khuyên 5-10 lớp của TrustGraph, và nó phân loại vật được mô tả chứ không phân loại chủ đề hay loại tri thức. `creativeWorkStatus = Obsolete` là chỗ đặt "tài liệu đã bị thay thế", đúng thứ ô `counter` của calibrated-judgment cần. Confidence: medium-high.

Suy luận:
- [inferred] Bỏ embedding thì phải có thứ khác trả chi phí lúc nạp để thu kho về vài chục ứng viên trước khi Laya đọc. Thứ đó chính là bản đồ Zero mô tả. Confidence: high. Reason: độ trễ đo được; Laya đọc từng cặp (câu hỏi, đoạn).
- [inferred] Phân loại theo nguồn gốc là thứ code đã biết miễn phí và chính xác; hỏi model điều đó chỉ tốn tiền và sai 48-72%. Quy tắc: không hỏi model điều code biết. Confidence: high. Reason: phép đo phân loại.
- [inferred] Category ngữ nghĩa do model gán là tầng kém tin nhất của bản đồ, kể cả với Jev. Vì vậy nó không được làm cổng chặn; chỉ dùng để mở rộng hoặc cộng điểm. Confidence: medium. Reason: bốn model đều yếu trên một bài phân loại có nội dung chồng lấn; chưa đo trên bộ category tách bạch.
- [inferred] Laya không nhanh hay chậm là vấn đề (7.6 ms/đoạn). Vấn đề là độ đúng zero-shot. Đổi sang TypeSafe không giải cái "chậm" mà Zero lo, và đưa dữ liệu ra khỏi máy. Confidence: high. Reason: đo tốc độ + Jev là API cloud.
- [inferred] Giới hạn ~20 lựa chọn của Laya buộc category phải là cây (mỗi tầng ≤ ~15-20 nhánh), không phải danh sách phẳng. Confidence: high. Reason: README token budget.
- [inferred] "RAG" ở đây là chữ R. Chữ G là người gọi (agent hoặc người). Confidence: high. Reason: Laya không sinh chữ; model sinh chỉ xuất hiện lúc nạp.
- [inferred] Thứ đang nổi lên không phải GraphRAG của Microsoft: bản đó dùng LLM rút entity + quan hệ ở mọi đoạn rồi viết tóm tắt cộng đồng, tốn kém và cần model mạnh. Nó gần hơn với chỉ mục dạng cây kiểu PageIndex (không vector, điều hướng theo cây mục lục) cộng một đồ thị link do code rút ra. Confidence: medium. Reason: mô tả PageIndex (github.com/VectifyAI/PageIndex); GraphRAG arXiv 2404.16130.

### Từ vựng
| Term | Nghĩa | Source | Status |
| --- | --- | --- | --- |
| Laya | model quyết định local, trả `choice/score/noul`, không sinh chữ | README `NandhaKishorM/laya` | existing |
| `noul` / `choice` / `score` | có/không; chọn một nhãn; mức trên thang | README Decision Primitives | existing |
| Jev | model judge của TypeSafe, chạy cloud | `judge_batch.status()` | existing |
| MiniCPM | họ model sinh chữ nhỏ chạy local của OpenBMB | README `OpenBMB/MiniCPM` | existing |
| source | một nguồn tri thức (repo, space Confluence, kênh Slack, hộp mail) | Zero: "hấp thụ source từ nhiều nguồn" | existing |
| item | đơn vị bên trong source mà CLI trả về | Zero: "trong source này có item gì" | existing |
| category | nhãn phân loại, xếp theo cây | Zero: "cateogry không chỉ list mà là theo cây" | existing |
| link | quan hệ source–source, item–item | Zero: "link vào nhau" | existing |
| bản đồ / map | cấu trúc dựng lúc nạp (cây + link + category) dẫn tới source | Zero: "một map bản đồ dẫn đến các source" | existing |
| ranker | bước Laya chấm và xếp item ứng viên | Zero: "dùng ranker là hợp" | existing |
| tiền xử lý | bước nạp và dựng bản đồ | Zero: "quan trọng nhất là ở bước tiền xử lý" | existing |
| BM25 | chỉ mục từ khoá, không vector | thuật ngữ IR chuẩn | existing |
| GraphRAG | RAG dùng đồ thị entity do LLM rút ra | Zero; Microsoft arXiv 2404.16130 | existing |

### Cảm biến và nguồn bằng chứng
- [observed] Bộ probe ở trên (hạng đáp án đúng, tỉ lệ phân loại đúng, ms/đoạn). Giới hạn: agent tự chọn câu hỏi và đáp án.
- [hypothesis] Cảm biến chính: 20-30 câu hỏi Zero viết trên repo + Confluence thật của anh, đáp án đúng do anh chỉ. Đo hạng đáp án đúng, thời gian mỗi câu, và hạng khi bật / tắt từng tầng bản đồ. Falsifier: câu hỏi của anh khác hẳn kiểu câu probe.
- [hypothesis] Cảm biến phía agent: agent dùng CLI để điền `evidence` / `standard` / `counter` của calibrated-judgment, và toạ độ trả về trỏ đúng dòng thật. Falsifier: agent vẫn phải tự grep lại.

### Quy tắc phương hướng
- [inferred] `away`: bất cứ bước nào gửi nội dung source private ra khỏi máy. Reason: "100% local" + "dữ liệu hỗn hợp".
- [inferred] `away`: Laya chấm cả kho mỗi câu hỏi. Reason: Zero bác ("quá khổ"); độ trễ đo xác nhận.
- [inferred] `away`: hỏi model điều code biết (nguồn gốc file, cây thư mục, cây trang, link tường minh). Reason: phép đo phân loại.
- [user-confirmed] `toward`: BM25. Evidence: "BM25 chấp nhận".
- [inferred] `toward`: cấu trúc và link lấy từ chính source bằng code. Reason: miễn phí, chính xác, đúng loại quan hệ Zero kể.

### Progress signals
- [hypothesis] Đáp án đúng trong top-5 trên bộ câu hỏi của Zero; mỗi câu vài giây trên máy này. Countermetric: mỗi tầng bản đồ phải làm hạng tốt hơn BM25 trơn trên cùng bộ; tầng nào không làm được thì bị bỏ.

### Guardrails
- [inferred] Chỉ mục dẫn xuất (item, link, BM25, category đã gán) chứa text private: chỉ nằm trong cache local, không bao giờ commit, nhất là vào repo public.
- [inferred] Không dùng xác suất `noul` thô làm ngưỡng "liên quan" (đã đo là bão hoà).
- [inferred] Category do model gán không được loại bỏ một ứng viên mà không có đường vòng.

### Correction triggers
- [hypothesis] Nếu BM25 + cây + link đã đạt progress signal trên dữ liệu của Zero, tầng category ngữ nghĩa bị hoãn và phải báo lại Zero, vì nó chạm vào mong muốn "category động".
- [observed] Đã kích hoạt trên bộ markdown: Laya zero-shot không hơn BM25 (top-1 14 vs 21). Xử lý theo lời Zero: bộ chấm TypeSafe cho giai đoạn thử / demo; Laya vào sau fine-tune. Phải đo lại trên bộ thật khi có Confluence.
- [observed] Đã kích hoạt khi build v1: mở rộng pool theo link không kéo thêm đáp án nào (38/40 cả hai) và làm top-1 của Jev từ 32 xuống 30 → tắt mặc định (`--links`). Tầng cây (đường dẫn + heading trong chỉ mục) thắng: top-5 30 → 32, top-10 32 → 35. Chi tiết: `docs/walks/inventio/walk.md`.

## Intent confirmation
- State: confirmed
- Confirmed-by: Zero
- Date: 2026-09-23
- Evidence: "được"
- Contract confirmed: trả lời ba mục đánh số của playback cuối; mục 1 là định nghĩa done (dòng 21), cộng thứ tự giá trị đã xác nhận trước đó ("1. đúng được ...").

## Candidate directions
### Bản đồ ba tầng, chia theo ai làm ra nó
- [hypothesis] Direction:
  1. Cây lấy từ chính source, bằng code: repo → thư mục → file → hàm/lớp (tree-sitter, như clerk); Confluence → space → cây trang → heading, label.
  2. Link rút bằng code: import, URL, link trang, mã ticket, @mention; và cầu nối repo ↔ Confluence qua định danh dùng chung (tên bảng, tên hàm, tên model dbt xuất hiện ở cả hai bên).
  3. Tầng ngữ nghĩa (category động, tiêu đề ngắn cho node, item mà source không tự đánh dấu): do model sinh làm lúc nạp, ghi ra file; chỉ dùng để mở rộng / cộng điểm.
  Lúc hỏi: BM25 lấy hạt giống → đi theo cây + link mở rộng → bộ chấm xếp lại ~30 ứng viên bằng một câu có/không có tiêu chí đúng/sai → trả đoạn gốc kèm toạ độ. Bộ chấm là một lựa chọn cấu hình: `typesafe` (thử, demo, source không nhạy cảm) hoặc `laya` (local, sau fine-tune bằng nhãn Jev). Metadata và link dùng từ vựng schema.org (`about`, `mentions`, `isPartOf`, `citation`, `creativeWorkStatus`).
- Supporting evidence: BM25 mạnh ở tầng thô; mọi model yếu ở phân loại; code miễn phí và đúng.
- Falsifier: trên bộ của Zero, đi theo link / cây không kéo thêm đáp án đúng nào mà BM25 bỏ sót.
- Cost against contract: cần viết connector + parser từng loại source; tầng 3 tốn thời gian nạp.
- Bearing: toward (tầng 1-2, bộ chấm Jev đã đo 21 → 32 top-1), unknown (tầng 3, Laya chưa fine-tune).
- Next discriminator: bộ câu hỏi của Zero trên repo + Confluence.

## Decisions (fork đang mở, chờ Zero)
- D1. [user-confirmed] Tool tự làm hết, local, không phụ thuộc coding agent; chạy `init` là ra bản đồ. Evidence: "coding agent vậy thì không phải local ... nó phụ thuộc vào coding agent đáng lý tool làm hết? ví dụ chạy init một cái nó ra luôn?" Hệ quả: thứ sinh category phải nằm trong tool. MiniCPM5-1B đã rớt cả gán lẫn đề xuất; 2B chưa đo.
- D2. [user-confirmed] Category chỉ để mở rộng / cộng điểm, không làm cổng lọc. Evidence: "D2 được".
- D3. [user-confirmed] TypeSafe làm model gán nhãn (thầy) để fine-tune Laya. Evidence: "D3 làm model gán nhãn". Điều kiện suy ra từ "100% local" + dữ liệu hỗn hợp: chỉ gán nhãn trên source public / không nhạy cảm. [inferred] — chờ Zero sửa nếu sai.
- D6. [user-confirmed] Chọn (c): `init` v1 chưa có trục loại tri thức, chỉ có metadata + chủ đề lấy từ cấu trúc; trục (b) là đích fine-tune. Evidence: "được", trả lời mục 3.
- D4. [open] Category động không cần model sinh: gom item theo từ đặc trưng dùng chung (đồ thị item–từ–item lấy từ chính chỉ mục BM25, rồi chia cộng đồng), tên category = vài từ đặc trưng nhất; model nhỏ (nếu có) chỉ đặt lại tên cho dễ đọc. [hypothesis]. Falsifier: Zero nhìn các cụm trên repo của anh mà không nhận ra chủ đề nào.
- D5. [inferred] NER tổng quát (`Minibase/NER-Standard`): chưa dùng. Lý do: loại entity (người / tổ chức / địa điểm) không phải thứ nối repo với Confluence; chỉ tiếng Anh; số đo tự báo, n=100. Thứ nối thật là định danh (tên bảng, hàm, model dbt, mã ticket): code rút được chính xác, rồi dùng làm từ điển để dò trong text Confluence. Falsifier: trên dữ liệu của Zero, phần lớn link đáng giá nằm ở văn xuôi không có định danh nào.
- D7. [user-confirmed] Bộ chấm có tuỳ chọn TypeSafe cho thử / demo; Laya fine-tune sau. Evidence: "để test tạm thời có option dùng type safe ai sau đó fine tune laya sau". Guardrail đi kèm [inferred]: chế độ TypeSafe gửi text ra cloud, nên chỉ bật cho source không nhạy cảm; Confluence ngân hàng phải qua kiểm tra chính sách trước khi bật.
- D8. [inferred] schema.org: dùng làm từ vựng link + metadata, không làm cây category. Chờ Zero sửa nếu anh định dùng nó theo cách khác.

## Direction acceptance
- State: accepted
- Acceptance basis: direct
- Accepted-by: Zero
- Date: 2026-09-23
- Evidence: "được" (trả lời mục 2: "Hướng đi: bản đồ ba tầng ... Anh trả lời chấp nhận / sửa.")
- Technical adjudicator and verdict: không áp dụng
- Direction accepted: bản đồ ba tầng ở "Candidate directions"; bộ chấm cấu hình được `typesafe` / `laya`; schema.org là từ vựng link + metadata; tầng 3 (category do model sinh) chưa vào v1 theo D6.
- Material unknowns carried forward: câu hỏi thật của Zero; Confluence (connector, chính sách gửi cloud); tầng cây / link có kéo thêm đáp án BM25 bỏ sót không; Laya sau fine-tune.
- First correction trigger: trên bộ markdown, mở rộng theo cây / link không thắng BM25 trơn → tắt mặc định, báo Zero.

## Critical paths
- Laya zero-shot yếu cả ở xếp hạng lẫn phân loại: [observed]; hệ quả: nếu dồn chất lượng lên Laya thì CLI thua BM25 miễn phí; proof sau này: hạng so với BM25 trơn trên bộ của Zero.
- Model sinh nhỏ (MiniCPM5-1B) dồn nhãn trên bài phân loại zero-shot: [observed]; hệ quả: category động có thể thành rác trông có vẻ gọn; proof sau này: bật / tắt tầng 3 trên bộ của Zero.
- Dữ liệu hỗn hợp + repo public: [user-confirmed]; hệ quả: rò text private qua cache hoặc commit; proof sau này: cache nằm ngoài repo, không có gì dẫn xuất trong git.

## Orientation checkpoints
### 2026-09-23 — contract, sau probe đầu
- Observed: Laya không sinh chữ; zero-shot kém; BM25 6/8 hạng 1; ~7.6 ms/đoạn.
- Next move: Zero trả lời Q1-Q4. → Đã trả lời.

### 2026-09-23 — playback hợp đồng
- Process stage: confirm orientation contract
- User-confirmed: CLI lên GitHub, lưu file local; không embedding, BM25 được; người đọc là cả agent lẫn người; repo + Confluence trước; dữ liệu hỗn hợp, repo private + public; tiền xử lý là trọng tâm; category dạng cây, có link.
- Observed: bốn model phân loại đều yếu (28 / 44 / 30 / 52%); MiniCPM5-1B dồn nhãn; SQLite FTS5 có `bm25()`.
- Inferred: bản đồ ba tầng chia theo ai làm; không hỏi model điều code biết; category ngữ nghĩa chỉ dùng để mở rộng; không phải GraphRAG kiểu Microsoft.
- Decided: chưa.
- Still open: thứ tự giá trị, định nghĩa done, D1-D3.
- Next move: Zero xác nhận hoặc sửa playback.
- Response: đúng / sửa / thiếu / dừng

### 2026-09-23 — bộ đo markdown, playback cuối
- Process stage: confirm orientation contract → decide
- User-confirmed: thứ tự giá trị (local cứng; đúng > nhanh > rẻ); TypeSafe là tuỳ chọn cho thử / demo, Laya fine-tune sau; đo tạm trên markdown; D1 tool tự làm, D2 mở rộng / cộng điểm, D3 Jev gán nhãn.
- Observed: 2.104 đoạn, 40 câu: BM25 top-1 21 / top-5 30; Laya ≤ 14 / 29; Jev 32 / 37, $0.033 cả mẻ.
- Inferred: TypeSafe cho demo là lựa chọn đo được; schema.org là từ vựng link, không phải cây category.
- Still open: xác nhận done; chấp nhận hướng; D6 (trục loại tri thức ở v1).
- Next move: Zero trả lời hai lượt riêng — Intent: đúng / sửa; Direction: chấp nhận / sửa.
- Response: đúng / sửa / thiếu / dừng

## Handoff guard
- Intent confirmation: complete
- Direction acceptance: complete
- Planning handoff: ready
- Implementation authority: "được" (2026-09-23), sau câu "Anh trả lời chấp nhận / sửa" cho hướng đi
- Orientation contract: sufficient
- Stranger test: pass — baseline có nguồn, đích và giá trị có trích dẫn, hướng và bằng chứng đo tách bạch, cái còn mở đều có cách xử lý.
- Remaining unknowns and disposition: đo lại trên Confluence → khi Zero đưa; chính sách gửi Confluence qua TypeSafe → Zero kiểm với ngân hàng; fine-tune Laya → sau khi có nhãn Jev.
