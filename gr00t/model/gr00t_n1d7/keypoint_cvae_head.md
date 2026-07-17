# Keypoint head: object-centric position prediction for the flat-point pipeline

Đi kèm data pipeline mới `keypoint_tracking_simple_pipeline.md` (repo `lerobot-convert`,
`experiment/test_keypoint_tracking_simple.py` / `convert_keypoint_tracking_simple.py`): thay vì
object-slot + active FSM (hysteresis, sticky role, mask-gate — `select_active_per_frame`), pipeline
mới track **N*K điểm phẳng** (N = `max_keypoint_objects`, K = `keypoints_per_object`) từ 1 init frame
duy nhất, mỗi điểm có **identity cố định suốt episode** — không còn khái niệm "role" bị gán lại giữa
chừng. Điều này loại bỏ 2 vấn đề đã gặp với thiết kế cũ, áp dụng cho **cả 4 mode**
(`default`/`tokens`/`share_dim`/`cvae`), không chỉ riêng CVAE:

- **Keypoint active bị overfit**: active giờ là mask tĩnh (`keypoint_valid`, không đổi theo frame),
  không còn gì để học/overfit — **không mode nào còn active decoder nữa** (`keypoint_active_decoder`
  đã bị xoá hoàn toàn khỏi codebase).
- **Permutation ambiguity theo TRỤC THỜI GIAN**: vì point k luôn là cùng 1 điểm vật lý suốt episode,
  không cần Chamfer set-matching giữa các timestep nữa — **mọi mode** giờ regress theo index qua 1 hàm
  dùng chung: `_compute_keypoint_position_loss(pred_kp, action_input)`.

## Point identity: anchor t=0 tại decoder (`tokens`/`cvae`)

Thứ tự K điểm trong 1 object KHÔNG well-defined giữa các episode: K điểm được chọn bằng
`farthest_point_sample` (`test_keypoint_tracking_simple.py`), bắt đầu từ 1 pixel random rồi greedy
chọn điểm xa nhất — index k chỉ nhất quán trong nội bộ 1 episode.

**Bài học collapse (đã gặp thực tế khi train)**: mọi cách gán target không có identity rõ ràng đều
có nghiệm suy biến dồn toàn bộ prediction về 1 điểm:
- *By-index trên target exchangeable* (không anchor): các điểm cùng 1 vật rắn chuyển động gần y hệt
  nhau ⇒ thứ hạng motion giữa chúng là nhiễu thuần túy theo từng window ⇒ target của token i là biến
  ngẫu nhiên trên cả vùng chuyển động ⇒ nghiệm Huber-tối-ưu của MỌI token là cùng 1 điểm trung tâm ⇒
  collapse.
- *Chamfer một chiều (pred→nearest target)*: dồn hết prediction lên đúng 1 điểm target bất kỳ cho
  loss ≈ 0 vì không ai bị phạt cho các target không được phủ ⇒ collapse.

**Thiết kế hiện tại — anchor t=0 tại decoder (ATM-style)**, cho `tokens`/`cvae`:
- **Token vào DiT = `keypoint_query_base` per-slot** (`[1, n_key, D]` — mỗi điểm 1 embedding học
  riêng, cộng thêm z_style ở cvae) — token KHÔNG mang tọa độ, nhưng mỗi slot tự phát triển attention
  riêng để track motion của điểm thứ i theo thứ tự emit của processor (motion-rank).
- **Identity vào ở decoder**: `keypoint_position_decoder(concat(h, keypoint_query_coord_encoder(
  anchor)))` — anchor = vị trí t=0 của từng điểm (`keypoint_target[:, 0]`, dữ liệu tracker có sẵn
  trong sample), encode bằng MLP nhỏ (2 → hidden → hidden). Decoder predict **`keypoint_horizon - 1`
  step TƯƠNG LAI (t=1..H-1)** — t=0 là input, không phải target.
- **`--keypoint-relative`**: target là displacement so với anchor thay vì tọa độ tuyệt đối —
  zero-centered, điểm đứng yên có target đúng bằng 0 (bài toán regress dễ hơn); viz/eval tự cộng
  anchor lại thành absolute.
- Loss quay về **by-index thuần**: mỗi slot có anchor riêng nên target là duy nhất theo input —
  không còn exchangeable, không cần Hungarian ở 2 mode này.

Hungarian (`Gr00tN1d7ActionHead._match_keypoints_hungarian`, bijection chặt qua
`scipy.linear_sum_assignment`, cost gộp cả horizon) giữ lại cho **`default`** —
`keypoint_match: "index" | "hungarian"`, match trong nội bộ từng object (không xuyên object — trục
object vẫn mang tín hiệu thật). `share_dim` luôn `"index"` (không có prediction đã decode để match
trước khi flow-matching loss chạy).

## Chọn `n_key` điểm: top-k theo motion (processor-side, `tokens`/`cvae`)

`keypoint_n_key` giờ là **selection thật ở processor** (không phải subsample loss như thiết kế cũ —
`_sample_keypoint_indices` đã xóa): mỗi training sample, `Gr00tN1d7Processor` chọn `n_key` điểm từ
tập điểm VALID theo quy tắc **2 tầng** (motion score = tổng displacement từng step trên cả horizon:
`diff = kp[1:] - kp[:-1]; motion = diff.norm(-1).sum(0)`; thiếu điểm valid thì cycle danh sách đã
rank):

1. Điểm **chuyển động thật** (score ≥ `_KEYPOINT_STATIC_MOTION_EPS = 0.05`, tọa độ [-1,1]): rank
   theo motion giảm dần — thứ hạng có nghĩa thật.
2. Điểm **gần đứng yên**: nối vào sau theo **flat index tăng dần**. Lý do bắt buộc: norm không âm
   nên jitter của tracker TÍCH LŨY trong score — điểm đứng yên vẫn có score > 0 nhưng thuần nhiễu,
   thứ hạng giữa chúng random theo từng window ⇒ nếu rank cả tầng này theo motion thì tập được chọn
   nhảy loạn giữa các step (video nhấp nháy, supervision đổi điểm vật lý liên tục). Index order thì
   ổn định tuyệt đối giữa các window.

Điểm chuyển động mang tín hiệu tương tác vật thể — được ưu tiên; window hoàn toàn tĩnh ⇒ chọn
`n_key` điểm valid đầu theo index, ổn định 100%. Chọn lọc **deterministic theo window** nên các
frame liên tiếp của 1 episode chọn tập gần như trùng nhau (video eval xem được, metric eval so sánh
được — không cần toggle riêng). Còn 1 nguồn flicker nhỏ đã biết: điểm có motion lởn vởn quanh
ngưỡng eps có thể đổi tầng giữa 2 window — bounded và hiếm. Lưu ý: motion score đọc từ **future
track (chính là label)** — hợp lệ để chọn cái gì được
supervise/viz, không có gì trong đó chạm vào action path. Emit `keypoint_target [H, n_key*2]` +
`keypoint_active_target [H, n_key]` (per-POINT valid, phân biệt với per-object `[H, N]` của chế độ
full-set qua last dim). Head append đúng `n_key` point token, decoder nhận anchor t=0 của đúng các
điểm đó, loss update đúng các điểm đó. `keypoint_n_key=None` = dùng cả N*K điểm (vẫn kiến trúc
anchored, chỉ là không chọn lọc).

## Inference / real robot: zero overhead

Nhờ mask 1 chiều (state/action không bao giờ attend vào point token), attention của các token
protected được tính y hệt như khi point token không tồn tại ⇒ `get_action_with_features` **chỉ append
point token khi `options["return_keypoints"]=True`** (eval/viz — anchor t=0 lấy từ `keypoint_target`
của chính eval batch, trainer truyền vào qua `action_input`). Trên real robot (mặc định không yêu
cầu keypoint): không token, không anchor, không tracker — `action_pred` giống hệt từng bit, sequence
còn NGẮN hơn code cũ (trước đây append 16 query token vô ích ở mọi denoising step). Anchor chỉ cần
khi muốn XEM keypoint prediction — và ở eval thì GT có sẵn.

## Tóm tắt 4 mode sau khi đồng bộ

| Mode | Vị trí decode position | Token layout | Pure readout? |
|---|---|---|---|
| `default` | hidden state của action token (mỗi step decode cả N*K điểm) | không thêm token | Có |
| `tokens` | decoder(concat(h_i, encoder(anchor_i))) → H-1 step tương lai | `n_key` (hoặc N*K) slot embedding riêng từng điểm | Có |
| `share_dim` | trực tiếp từ `action_decoder` (fold vào flow-matching) | không thêm token, widen action channel | Không |
| `cvae` | như `tokens` + z_style cộng vào point token | như `tokens` | Có (z_style không leak nhờ mask 1 chiều) |

`tokens`/`cvae` giờ dùng chung kiến trúc point-token (khác nhau đúng phần CVAE encoder + z_style);
`default`/`share_dim` giữ layout cũ theo-step, predict full N*K theo index.

## `share_dim`: đổi từ fold active-flag sang fold position

Trước đây `share_dim` chỉ fold **active flag** (N chiều) vào flow-matching vì position lúc đó cần
Chamfer (permutation ambiguity, không well-posed để fold trực tiếp theo channel cố định). Giờ point
identity đã cố định, fold **position** (N*K*2 chiều — rộng hơn nhiều, từ N lên N*K*2) là lựa chọn hợp
lý hơn (active không còn gì để fold — nó là hằng số đã biết, không phải bài toán dự đoán thật). Xem
`_share_dim_position_targets` (thay cho `_share_dim_active_targets` cũ) và
`action_dim = max_action_dim + max_keypoint_objects * keypoints_per_object * 2`.

## Vì sao `cvae` cần thêm CVAE (khác biệt so với 3 mode kia)

Future keypoint có thể **đa modal** (vật nào sẽ được thao tác trước, theo hướng nào) — regress
thẳng (MSE/Huber, như 3 mode kia làm) trên target đa modal sẽ hội tụ về giá trị trung bình (mờ, sai).
CVAE tách: encoder q(z_style | future thật, condition) chỉ chạy lúc train (thấy label thật) để nén
"mode nào đã xảy ra" vào 1 latent nhỏ; decoder p(keypoints | z_style, context) học tái tạo lại đúng
future đó từ z_style + context. Lúc infer không có label thật → z_style mặc định = 0 (mean của prior
N(0,I) mà KL loss huấn luyện cho posterior tiến gần tới).

## Pipeline (`cvae` mode)

```
Processor: chọn top-n_key điểm valid chuyển động nhiều nhất ──► keypoint_target [H, n_key, 2]
           (motion score trên window, deterministic)             keypoint_active_target [H, n_key]
                    ┌─ TRAIN ONLY ───────────────────────────────────────┐
                    │  keypoint_target (label thật, n_key điểm đã sample)│
                    │        │                                          │
                    │        ▼                                          │
condition_token ───►│  [cls] [condition] [label_t0..t15] ──► SelfAttn ──►│── mu, logvar (clamp ±10)
(vlm pooled, hoặc   │                                          Transformer│      │
 state)             │                                                    │  reparameterize
                    └────────────────────────────────────────────────────┘      │
                                                                                  ▼
                                                                              z_style
                                                                     (train: sample; infer: zeros)
                                                                                  │
state ──┐                                                                        │
        ├─► sa_embs ──► DiT (mask 1 chiều: sa_embs KHÔNG thấy point token) ◄──────┘
action ─┘        │                                                        │
                  ▼                                     n_key point token: │
             action_decoder ──► action_pred              query_base[i] (per-slot) + z_style
        (không đổi dù bật/tắt keypoint head;                       │
         robot: point token không được append)                     ▼ h (context chung)
                     keypoint_position_decoder(concat(h, coord_encoder(anchor t=0 mỗi điểm)))
                                          ──► H-1 step tương lai (absolute hoặc relative)
                                                                   │
                          reconstruction loss (Huber by-index — well-posed vì identity gắn với
                          anchor input; --keypoint-relative: target = displacement so với anchor)
```

## Quyết định thiết kế quan trọng (riêng cho `cvae`)

1. **Condition = VLM output (pooled), không phải `state`.**
   Encoder cần "biết" ít nhất bằng những gì decoder biết, để z chỉ phải mã hoá phần thực sự ngẫu
   nhiên (mode nào xảy ra) chứ không phải bù cho context yếu. Dùng `state` (chỉ proprioception,
   không thấy scene) sẽ ép z phải cõng cả thông tin thị giác → z quá "nặng", decoder ỷ lại z, bỏ qua
   conditioning thật → hỏng khi inference set z=0 (posterior collapse theo hướng xấu). Cấu hình qua
   `keypoint_cvae_condition: "vlm" | "state"` (mặc định `"vlm"`).

2. **z_style nối vào point tokens, KHÔNG nối vào `sa_embs`.**
   `sa_embs` (state+action) được dùng chung để `action_decoder` ra `action_pred`. Nếu z_style (vốn
   được encoder tính từ **label thật** lúc train) lọt vào `sa_embs`, `action_pred` sẽ "nhìn trộm"
   tương lai lúc train nhưng lúc infer z=0 (không có gì) → lệch train/inference, hỏng action một
   cách âm thầm (train loss trông tốt giả tạo). Thay vào đó, z_style được cộng vào các point token
   (cơ chế `"tokens"` mode có sẵn) — các token này bị `_keypoint_self_attention_mask` chặn 1 chiều:
   state/action **không bao giờ** attend ngược lại point token, nên `action_pred` **provably
   unaffected**, đúng guarantee mà `"default"`/`"tokens"` mode đã có.

3. **Encoder và decoder thấy CÙNG tập `n_key` điểm đã sample.**
   Processor sample 1 lần mỗi sample; cả CVAE encoder (label = future của đúng các điểm đó) lẫn
   decoder (predict + loss trên đúng các điểm đó) đều làm việc trên cùng tập — posterior và decoder
   thống nhất về "future nào đang được mô tả". `keypoint_label_step_embed` có input dim `n_key*2`
   nên đổi `keypoint_n_key` là đổi shape checkpoint (validation trong config đã chặn dùng
   `keypoint_n_key` với `default`/`share_dim`).

4. **z_dim nhỏ (`keypoint_style_dim=16`) + zero-init `keypoint_style_head`.**
   z phải là bottleneck thật sự — nếu đủ lớn để tái tạo losslessly cả trajectory, decoder sẽ học bỏ
   qua conditioning thật, chỉ dựa z (tốt lúc train vì z có label, nhưng vỡ lúc infer vì z=0).
   Zero-init khiến encoder khởi động ở đúng mu=0, logvar=0 (= prior), để z=0 lúc infer khớp với
   những gì decoder thấy ngay từ đầu training, không phải một điểm ngẫu nhiên decoder chưa từng gặp.

   **Rủi ro này ĐÃ xảy ra thực tế** (quan sát: `keypoint_kl_loss` tăng mạnh trong training): với
   `keypoint_kl_weight=0.01` quá yếu, encoder cứ đẩy `mu` xa prior — mỗi nat thông tin trong z mua
   được nhiều reconstruction hơn giá KL phải trả, đặc biệt sau khi bỏ coord anchor (bài toán khó hơn
   → thông tin "motion ở đâu" trong z càng có giá). Decoder ỷ lại z; inference (z=0) rơi ra ngoài
   phân phối z từng thấy → hỏng. **Đối sách**:
   - Tăng `--keypoint-kl-weight` lên `0.1` (thử trước), chưa đủ thì `0.5`–`1.0`. KL nên ổn định ở
     mức vài nat, không leo tuyến tính.
   - Có thể giảm `--keypoint-style-dim` 16 → 8 (siết bottleneck).
   - `forward()` ở EVAL giờ dùng **z=0** (đúng như inference) thay vì sample từ posterior — nên
     `eval_keypoint_loss` giờ đo thẳng chất lượng deployment; **gap train↔eval keypoint_loss chính
     là thước đo mức độ decoder ỷ lại z**. KL vẫn tính từ posterior như cũ.

## Config (`Gr00tN1d7Config` / `FinetuneConfig`)

| Field | Default | Áp dụng | Ghi chú |
|---|---|---|---|
| `keypoint_head_mode` | `"default"` | mọi mode | `{"default","tokens","share_dim","cvae"}` |
| `keypoint_loss_weight` | 1.0 | mọi mode | trọng số Huber position loss |
| `static_keypoint_weight` | 0.0 | mọi mode | trọng số cho slot padding (valid=0) |
| `keypoint_style_dim` | 16 | `cvae` | chiều z_style |
| `keypoint_kl_weight` | 0.01 | `cvae` | trọng số KL(q\|\|N(0,I)) — beta-VAE style |
| `keypoint_cvae_condition` | `"vlm"` | `cvae` | `"vlm"` (khuyến nghị) hoặc `"state"` |
| `keypoint_cvae_encoder_layers` | 2 | `cvae` | số layer self-attention của encoder |
| `keypoint_cvae_encoder_heads` | 4 | `cvae` | phải chia hết `input_embedding_dim` |
| `keypoint_n_key` | `None` | `tokens`/`cvae` | processor sample n_key điểm valid mỗi sample; `None` = dùng cả N*K; đổi giá trị = đổi shape checkpoint |
| `keypoint_match` | `"index"` | `default` | `"hungarian"` = bijection trong nội bộ 1 object; `tokens`/`cvae` by-index theo anchor, `share_dim` luôn `"index"` |
| `keypoint_relative` | `False` | `tokens`/`cvae` | target = displacement so với anchor t=0 (zero-centered, điểm tĩnh = 0); đổi giá trị = checkpoint không tương thích về semantics |

`keypoint_active_loss_weight` đã bị **xoá** (không còn field này trong config) — không mode nào còn
active loss để weight nữa.

## Loss & log keys

- `keypoint_loss` — reconstruction Huber, trọng số `keypoint_loss_weight`, **mọi mode** đều report
  key này (kể cả `share_dim`, tính trực tiếp trên velocity thay vì qua decoder riêng).
- `keypoint_kl_loss` — chỉ `cvae`, trọng số `keypoint_kl_weight`. `trainer.py` chỉ track/log key này
  khi `keypoint_head_mode == "cvae"`.
- `keypoint_active_loss` — **đã bị xoá hoàn toàn**, không mode nào emit key này nữa.

## Data pipeline (dataset trên đĩa): KHÔNG cần đổi

Dataset vẫn là schema cũ (`keypoint_2d` + `keypoint_valid` per-object trong `meta/modality.json`).
Việc sample `n_key` và emit per-point valid xảy ra hoàn toàn trong `Gr00tN1d7Processor` lúc
training — không cần convert lại data. `keypoint_active_target` emit ra batch giờ có 2 layout,
phân biệt bằng last dim: `[H, N]` per-object (khi `keypoint_n_key=None`) hoặc `[H, n_key]` per-point
(khi set); `_compute_keypoint_position_loss` và trainer viz đều tự nhận biết.

## Checkpoint migration (`gr00t/model/gr00t_n1d7/setup.py`)

Load từ checkpoint cũ sẽ tự động:
- Discard (không lỗi) mọi weight retired: `keypoint_decoder.*`, `keypoint_active_decoder.*`,
  `keypoint_query_embedding.*`, `keypoint_rank_embedding.*` (bị thay bằng `keypoint_query_base` +
  `keypoint_query_coord_encoder` phục vụ decoder anchored).
- Re-init tường minh (`_reinit_missing_keypoint_params` — tránh fast-init memory rác) mọi keypoint
  param missing HOẶC mismatch shape (vd `keypoint_position_decoder` đổi cả input lẫn output dim khi
  chuyển sang anchored: `hidden*2 → (H-1)*2`).
- Với `share_dim`: splice logic (`action_dim_mismatched`) không đổi.

## Inference

`get_action_with_features`: point token **chỉ được append khi `options["return_keypoints"]=True`**
(anchor t=0 lấy từ `keypoint_target[:, 0]` của eval batch; robot không bao giờ cần). `share_dim`
đọc keypoint trực tiếp từ `actions[..., real_action_dim:]` (slice về `keypoint_horizon` — phần sau
không được supervise); `default` decode từ action-token hidden state. `cvae`: z_style mặc định
`zeros`; `options={"keypoint_style_sample": True}` để sample z_style ~ N(0,I) (xem đa dạng future)
— không ảnh hưởng `action_pred` dù chọn cách nào. `keypoint_pred` giữ contract 5-D
`[B, H, groups, points_per_group, 2]` cho viz (mọi mode đều đúng `H = keypoint_horizon` step):
`default`/`share_dim` group theo object `[B,H,N,K,2]`; `tokens`/`cvae` mỗi điểm là 1 group
`[B,H,n_key,1,2]`, step 0 là anchor prepend lại. **`keypoint_active_pred` đã xóa** — không mode nào
predict "active"; viz weight cả 2 panel (GT lẫn pred) bằng GT valid mask, nhờ đó window không có
điểm valid tự động trống ở cả 2 panel.

## Tham chiếu code

- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`:
  `_compute_keypoint_position_loss` (dùng chung `default`/`tokens`/`cvae`; slice t=1.. + relative
  cho anchored modes), `_keypoint_anchor` (anchor t=0 từ `keypoint_target`),
  `_decode_keypoint_positions` (concat(h, coord_encoder(anchor)) → H-1 step),
  `_match_keypoints_hungarian` (`default` + `keypoint_match="hungarian"` only),
  `_share_dim_position_targets`, `_append_keypoint_queries`
  (point token: slot embedding riêng từng điểm + z_style), `_init_keypoint_cvae_modules`,
  `_keypoint_condition_token`, `_encode_keypoint_style` (logvar clamp ±10),
  `_sample_keypoint_style`, `_compute_keypoint_kl_loss`.
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`: chọn `keypoint_n_key` điểm valid (2 tầng
  motion/index) mỗi sample.
- `gr00t/configs/model/gr00t_n1d7.py`: docstring đầy đủ trong `keypoint_head_mode`; validation
  `keypoint_n_key` (chỉ `tokens`/`cvae`, 1..N*K) + `keypoint_relative` (chỉ `tokens`/`cvae`).
- `gr00t/model/gr00t_n1d7/setup.py`: `_create_model` — checkpoint migration + re-init tường minh.
- `gr00t/experiment/trainer.py`: log `keypoint_kl_loss` khi cvae; viz broadcast per-object →
  per-point khi cần.
