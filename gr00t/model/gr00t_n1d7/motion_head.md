# Motion-keypoint head: VLM-backbone position prediction + human/robot OT alignment

Thiết kế hiện tại cho human/robot co-training — thay thế hoàn toàn keypoint head cũ ở phía Action
Head (đã xoá khỏi codebase). Xem `method_motion_keypoint_ot_v2.md` ở gốc repo cho lý do thiết kế đầy
đủ (tại sao chuyển sang phía VLM backbone, tại sao hợp nhất với OT alignment); doc này ghi lại chi
tiết triển khai + thừa kế trực tiếp phần phân tích point-identity từ thiết kế cũ.

## Kiến trúc

`num_motion_tokens` token học được (`Qwen3Backbone.motion_query_tokens`,
`gr00t/model/modules/qwen3_backbone.py`) được nối vào CUỐI sequence input của chính VLM backbone —
đi qua toàn bộ transformer layers của backbone cùng ảnh/ngôn ngữ, không phải DiT. Vì nối ở cuối và
backbone là causal LM, các token này attend được vào mọi token ảnh/ngôn ngữ phía trước, nhưng không
gì attend ngược lại được vào chúng — pure readout đối với action path, có được miễn phí từ causal
masking thay vì cần tự viết attention mask một chiều như thiết kế cũ.

`MotionHead` (`gr00t/model/gr00t_n1d7/motion_head.py`) đọc hidden state sau-backbone của các token
này, decode vị trí tương lai bằng 1 forward pass thuần feedforward (KHÔNG cần flow-matching rollout
như thiết kế cũ — motion token không nằm trong DiT nên không có gì để denoise).

## OT alignment: match trên motion, align trên backbone feature (2 không gian TÁCH BIỆT)

`enable_ot_align` không align trực tiếp `motion_pooled_features` (mean qua trục token,
`motion_pool="mean"`) với chính nó. Thay vào đó, tách rõ 2 vai trò (`gr00t/model/modules/
optimal_transport.py`'s `sinkhorn_ot_loss(..., target_robot=, target_human=)`):

- **Không gian matching** = `motion_pooled_features` — tín hiệu embodiment-invariant "chuyển động
  này là gì", chỉ được train bởi `motion_loss` (sạch, không bị ảnh hưởng bởi mục tiêu alignment —
  xem stop-gradient bên dưới). Dùng để tính transport plan `P* = Sinkhorn(cost(motion_robot,
  motion_human))`.
- **Không gian align (target thật)** = `backbone_pooled_features` — masked-mean-pool của
  `action_outputs["backbone_features"]` (chính là hidden state đã qua `vlln`/`vl_self_attention`,
  cái THẬT SỰ feed vào DiT qua cross-attention — xem `Gr00tN1d7ActionHead.process_backbone_output`),
  tính trong `Gr00tN1d7.forward()` (hàm `_masked_mean_pool`). `L_align = Σ P*_ij · d(backbone_robot_i,
  backbone_human_j)` — motion quyết định "ai khớp ai", nhưng lực kéo tác động lên đúng representation
  ảnh hưởng đến action prediction, không phải lên motion-token.

**Stop-gradient trên `P*` (mặc định `stopgrad_plan=True`)**: nếu không chặn, gradient của `L_align`
sẽ chảy ngược qua `P*(motion_pooled_features)` vào motion-token encoder — nghĩa là motion-token giờ
chịu ảnh hưởng gián tiếp từ mục tiêu "làm sao cho backbone dễ align hơn", phá vỡ tính "sạch" (chỉ
phục vụ `motion_loss`) vốn là lý do chọn nó làm tín hiệu dẫn đường. Detach `P*` giữ 2 không gian tách
bạch hoàn toàn: gradient của `L_align` chỉ chảy vào `backbone_pooled_features`/backbone qua cost
term, không chảy ngược vào motion-token qua plan. Tương đương thực hành chuẩn trong OT-based domain
adaptation (vd. DeepJDOT): plan được coi là cố định tại cost hiện tại mỗi bước, không lan truyền qua.

**Tách kiến trúc rõ**: `motion_token_features` (`hidden[:, real_len:]`) và `backbone_features`
(`hidden[:, :real_len]`) là 2 lát cắt KHÔNG chồng lấn của cùng 1 hidden sequence
(`gr00t/model/modules/qwen3_backbone.py`'s `forward()`) — chia sẻ tham số backbone (cùng 1
transformer xử lý 1 sequence chung) nhưng không bao giờ chia sẻ vị trí token/feature vector.

**Diagnostic bắt buộc trước khi tin kết quả** (`Gr00tTrainer._log_domain_alignment_viz`): vì
correspondence tìm được ở motion-space không tự động đảm bảo hợp lý ở backbone-space (giả định ngầm
2 không gian "đồng bộ" về ngữ nghĩa), so sánh k-nearest-neighbor structure giữa 2 không gian
(`{prefix}/motion_backbone_neighbor_agreement`, cùng 1 tập sample) — thấp nghĩa là 2 không gian
không đồng ý "ai giống ai", correspondence mượn từ motion-space có thể không phù hợp áp cho
backbone-space dù loss OT vẫn giảm. t-SNE scatter + KNN domain-composition + per-domain variance
(`{prefix}/backbone_domain_*`) chạy trên **backbone feature** — target align thật — để trả lời đúng
câu hỏi "robot/human feature có thực sự bị kéo lại gần nhau không", không phải câu hỏi về
motion-space. `is_human` là khái niệm tầng data/Trainer, cố tình không đưa vào forward().

## Point identity: anchor t=0 tại decoder (thừa kế nguyên vẹn từ thiết kế cũ)

Thứ tự K điểm trong 1 object KHÔNG well-defined giữa các episode: điểm được chọn bằng
`farthest_point_sample` (pipeline convert bên ngoài), bắt đầu từ 1 pixel random rồi greedy chọn
điểm xa nhất — index k chỉ nhất quán trong nội bộ 1 episode.

**Bài học collapse (đã gặp thực tế khi train thiết kế cũ)**: mọi cách gán target không có identity
rõ ràng đều có nghiệm suy biến dồn toàn bộ prediction về 1 điểm:
- *By-index trên target exchangeable* (không anchor): các điểm cùng 1 vật rắn chuyển động gần y hệt
  nhau ⇒ thứ hạng motion giữa chúng là nhiễu thuần túy theo từng window ⇒ target của token i là biến
  ngẫu nhiên trên cả vùng chuyển động ⇒ nghiệm Huber-tối-ưu của MỌI token là cùng 1 điểm trung tâm ⇒
  collapse.
- *Chamfer một chiều (pred→nearest target)*: dồn hết prediction lên đúng 1 điểm target bất kỳ cho
  loss ≈ 0 vì không ai bị phạt cho các target không được phủ ⇒ collapse.

**Thiết kế hiện tại — anchor t=0 tại decoder (ATM-style)**:
- Token vào backbone = `motion_query_tokens` per-slot (`[1, num_motion_tokens, D]` — mỗi điểm 1
  embedding học riêng) — token KHÔNG mang tọa độ, nhưng mỗi slot tự phát triển attention riêng để
  track motion của điểm thứ i theo thứ tự emit của processor (motion-rank).
- **Identity vào ở decoder**: `motion_position_decoder(concat(h, motion_coord_encoder(anchor)))` —
  anchor = vị trí t=0 của từng điểm (`keypoint_target[:, 0]`, dữ liệu tracker có sẵn trong sample),
  encode bằng MLP nhỏ (2 → hidden → hidden). Decoder predict **`motion_horizon - 1` step TƯƠNG LAI**
  (t=1..H-1) — t=0 là input, không phải target.
- `motion_relative=True`: target là displacement so với anchor thay vì tọa độ tuyệt đối —
  zero-centered, điểm đứng yên có target đúng bằng 0 (bài toán regress dễ hơn); viz/eval tự cộng
  anchor lại thành absolute (`Gr00tN1d7.forward`).
- Loss là **by-index thuần**: mỗi slot có anchor riêng nên target là duy nhất theo input — không còn
  exchangeable, không cần Hungarian/Chamfer matching (khác thiết kế `default` cũ, vốn có tuỳ chọn
  Hungarian — mode đó không còn tồn tại).

## Chọn `num_motion_tokens` điểm: top-k theo motion (processor-side)

`num_motion_tokens` là **selection thật ở processor**: mỗi training sample, `Gr00tN1d7Processor`
chọn `num_motion_tokens` điểm từ tập điểm VALID theo quy tắc **2 tầng** (motion score = tổng
displacement từng step trên cả horizon: `diff = kp[1:] - kp[:-1]; motion = diff.norm(-1).sum(0)`;
thiếu điểm valid thì cycle danh sách đã rank):

1. Điểm **chuyển động thật** (score ≥ ngưỡng noise floor, toạ độ `[-1,1]`): rank theo motion giảm
   dần — thứ hạng có nghĩa thật.
2. Điểm **gần đứng yên**: nối vào sau theo **flat index tăng dần** (ổn định tuyệt đối giữa các
   window, tránh flicker khi 2 điểm gần ngưỡng noise floor đổi thứ hạng ngẫu nhiên).

Điểm chuyển động mang tín hiệu tương tác vật thể — được ưu tiên; window hoàn toàn tĩnh ⇒ chọn
`num_motion_tokens` điểm valid đầu theo index, ổn định 100%. `num_motion_tokens=None` = dùng cả
N*K điểm (không chọn lọc).
