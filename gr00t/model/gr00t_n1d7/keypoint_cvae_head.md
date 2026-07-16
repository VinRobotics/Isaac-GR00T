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

## Permutation invariance: identity đến từ input, không phải từ loss

Thứ tự K điểm trong 1 object KHÔNG well-defined giữa các episode: K điểm được chọn bằng
`farthest_point_sample` (`test_keypoint_tracking_simple.py`), bắt đầu từ 1 pixel random rồi greedy
chọn điểm xa nhất — index k chỉ nhất quán trong nội bộ 1 episode. Có 2 cách xử lý, dùng cho các mode
khác nhau:

1. **`tokens`/`cvae` — query-conditioned point tokens (cách chính, ATM-style)**: mỗi điểm được cấp 1
   token = `keypoint_query_base` (base embedding học chung) + `keypoint_query_coord_encoder`(vị trí
   t=0 của điểm đó, lấy từ `keypoint_target[:, 0]` — dữ liệu tracker đã có sẵn trong sample). Mỗi
   token decode **toàn bộ quỹ đạo `keypoint_horizon` step của chính điểm đó**
   (`keypoint_position_decoder` output `keypoint_horizon*2` mỗi token). Identity gắn với **tọa độ
   input** chứ không phải slot index ⇒ hoán vị điểm ⇒ prediction/target hoán vị theo ⇒ Huber theo
   index tự động **permutation-invariant trên toàn bộ tập điểm** — không cần Chamfer/Hungarian, không
   rủi ro collapse. Processor còn sample điểm theo **thứ tự ngẫu nhiên** mỗi lần, nên model không thể
   bám vào vị trí token. (Đây cũng là câu trả lời cho "escnn?": escnn làm equivariance cho nhóm
   Euclide E(2)/E(3) — xoay/lật ảnh — không phải nhóm hoán vị S_n trên tập điểm; công cụ đúng cho
   S_n là set/transformer không đánh số token + identity từ input, chính là thiết kế này.)
2. **`default` — `keypoint_match: "index" | "chamfer"`**: mode duy nhất còn regress theo enumeration
   tuỳ ý của converter. `"chamfer"` match target về predicted gần nhất trong nội bộ 1 object (cost
   gộp cả horizon, 1 phép gán cố định mỗi sample/object) — xem
   `Gr00tN1d7ActionHead._match_keypoints_chamfer`; one-directional nên vẫn còn rủi ro collapse trùng
   lặp. `share_dim` luôn `"index"` (không có prediction đã decode để match trước khi flow-matching
   loss chạy).

## Sampling `n_key` điểm (processor-side, `tokens`/`cvae`)

`keypoint_n_key` giờ là **sampling thật ở processor** (không phải subsample loss như thiết kế cũ —
`_sample_keypoint_indices` đã xóa): mỗi training sample, `Gr00tN1d7Processor` sample `n_key` điểm từ
tập điểm VALID (loại padding slot; lấy without replacement khi đủ, with replacement khi thiếu), emit
`keypoint_target [H, n_key*2]` + `keypoint_active_target [H, n_key]` (per-POINT valid, phân biệt với
per-object `[H, N]` của chế độ full-set qua last dim). Head append đúng `n_key` point token, predict
đúng `n_key` điểm, loss update đúng các điểm đó. `keypoint_n_key=None` = dùng cả N*K điểm (vẫn
query-conditioned, chỉ là không sample).

## Inference / real robot: zero overhead

Nhờ mask 1 chiều (state/action không bao giờ attend vào point token), attention của các token
protected được tính y hệt như khi point token không tồn tại ⇒ `get_action_with_features` **chỉ append
point token khi `options["return_keypoints"]=True`** (eval/viz — query lấy từ GT của eval batch).
Trên real robot (mặc định không yêu cầu keypoint): không cần vị trí keypoint hiện tại, không tracker,
không token thừa — `action_pred` giống hệt từng bit, sequence còn NGẮN hơn code cũ (trước đây append
16 query token vô ích ở mọi denoising step).

## Tóm tắt 4 mode sau khi đồng bộ

| Mode | Vị trí decode position | Token layout | Pure readout? |
|---|---|---|---|
| `default` | hidden state của action token (mỗi step decode cả N*K điểm) | không thêm token | Có |
| `tokens` | point token (mỗi token decode cả quỹ đạo H step của 1 điểm) | `n_key` (hoặc N*K) point token, query = t0 coord | Có |
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
Processor: sample n_key điểm valid (random order, mỗi sample) ──► keypoint_target [H, n_key, 2]
                                                                   keypoint_active_target [H, n_key]
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
             action_decoder ──► action_pred              base + coord_enc(t0 của điểm) + z_style
        (không đổi dù bật/tắt keypoint head;                       │
         robot: point token không được append)                     ▼
                                       keypoint_position_decoder (mỗi token ──► quỹ đạo H step)
                                                                   │
                              reconstruction loss (Huber theo index — well-posed vì identity
                              đã gắn với coord input; mask theo per-point valid)
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

   **Rủi ro cần theo dõi thực nghiệm** (chưa validate): với `keypoint_kl_weight=0.01` khá nhỏ,
   posterior có thể học `logvar` rất nhỏ → z gần deterministic → decoder ỷ lại z thay vì học từ
   scene qua cross-attention → train loss thấp giả tạo, inference (z=0) kém. Theo dõi
   `keypoint_kl_loss` qua training; nếu nghi ngờ, so sánh eval với `keypoint_style_sample=False` (z=0)
   vs ép z=mu thật trên cùng batch.

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
| `keypoint_match` | `"index"` | `default` | `"chamfer"` = match trong nội bộ 1 object; `tokens`/`cvae` không cần (identity từ coord input), `share_dim` luôn `"index"` |

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
- Discard (không lỗi) mọi weight retired: `keypoint_decoder.*`, `keypoint_active_decoder.*`, và giờ
  thêm `keypoint_query_embedding.*` (bị thay bằng `keypoint_query_base` +
  `keypoint_query_coord_encoder`).
- Re-init tường minh (`_reinit_missing_keypoint_params` — tránh fast-init memory rác) mọi keypoint
  param missing HOẶC mismatch shape (vd `keypoint_position_decoder` đổi output dim khi chuyển sang
  point token).
- Với `share_dim`: splice logic (`action_dim_mismatched`) không đổi.

## Inference

`get_action_with_features`: point token **chỉ được append khi `options["return_keypoints"]=True`**
(query coords lấy từ `keypoint_target[:, 0]` của eval batch; robot không bao giờ cần). `share_dim`
đọc keypoint trực tiếp từ `actions[..., real_action_dim:]`; `default` decode từ action-token hidden
state. `cvae`: z_style mặc định `zeros`; `options={"keypoint_style_sample": True}` để sample
z_style ~ N(0,I) (xem đa dạng future) — không ảnh hưởng `action_pred` dù chọn cách nào.
`keypoint_pred` giữ contract 5-D `[B, T, groups, points_per_group, 2]` cho viz: `default`/`share_dim`
group theo object `[B,T,N,K,2]`; `tokens`/`cvae` mỗi điểm là 1 group `[B,T,n_key,1,2]`.
`keypoint_active_pred` trả hằng số 1 ở mọi mode, chỉ để code overlay cũ chạy không cần sửa.

## Tham chiếu code

- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`:
  `_compute_keypoint_position_loss` (dùng chung `default`/`tokens`/`cvae`),
  `_match_keypoints_chamfer` (`default` + `keypoint_match="chamfer"`),
  `_share_dim_position_targets`, `_append_keypoint_queries` + `_keypoint_query_coords`
  (point token: base + coord encoder + z_style), `_init_keypoint_cvae_modules`,
  `_keypoint_condition_token`, `_encode_keypoint_style` (logvar clamp ±10),
  `_sample_keypoint_style`, `_compute_keypoint_kl_loss`.
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`: sampling `keypoint_n_key` điểm valid mỗi sample.
- `gr00t/configs/model/gr00t_n1d7.py`: docstring đầy đủ trong `keypoint_head_mode`; validation
  `keypoint_n_key` (chỉ `tokens`/`cvae`, 1..N*K).
- `gr00t/model/gr00t_n1d7/setup.py`: `_create_model` — checkpoint migration + re-init tường minh.
- `gr00t/experiment/trainer.py`: log `keypoint_kl_loss` khi cvae; viz broadcast per-object →
  per-point khi cần.
