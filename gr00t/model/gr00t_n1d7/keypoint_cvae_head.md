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
- **Permutation ambiguity trong Chamfer loss**: vì point k luôn là cùng 1 điểm vật lý suốt episode,
  không cần Chamfer set-matching nữa — **mọi mode** giờ regress trực tiếp theo index qua 1 hàm dùng
  chung: `_compute_keypoint_position_loss(pred_kp, action_input, key_indices=None)`.

## Tóm tắt 4 mode sau khi đồng bộ

| Mode | Vị trí decode position | Có decoder riêng? | Pure readout? |
|---|---|---|---|
| `default` | hidden state của action token | Có (`keypoint_position_decoder`) | Có |
| `tokens` | hidden state của dedicated query token | Có (`keypoint_position_decoder`) | Có |
| `share_dim` | trực tiếp từ `action_decoder` (fold vào flow-matching) | **Không** — không có `keypoint_position_decoder` | Không (như trước, chỉ là active→position) |
| `cvae` | hidden state của dedicated query token + z_style | Có (`keypoint_position_decoder`) | Có (z_style không leak nhờ query-token routing) |

`default`/`tokens` gần như giống hệt nhau về mặt loss bây giờ (chỉ khác chỗ decode), vì cả 2 đều mất
phần active decoder — khác biệt kiến trúc thật sự chỉ còn giữa chúng và `share_dim`/`cvae`.

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
                    ┌─ TRAIN ONLY ───────────────────────────────────────┐
                    │  keypoint_target (label thật, full N*K)            │
                    │        │                                          │
                    │        ▼                                          │
condition_token ───►│  [cls] [condition] [label_t0..t15] ──► SelfAttn ──►│── mu, logvar
(vlm pooled, hoặc   │                                          Transformer│      │
 state)             │                                                    │  reparameterize
                    └────────────────────────────────────────────────────┘      │
                                                                                  ▼
                                                                              z_style
                                                                     (train: sample; infer: zeros)
                                                                                  │
state ──┐                                                                        │
        ├─► sa_embs ──► DiT (self-attn 1 chiều: sa_embs KHÔNG thấy z_style) ◄─────┘ (+ vào keypoint
action ─┘        │                                                                   query tokens)
                  ▼
             action_decoder ──► action_pred (không đổi dù bật/tắt keypoint head)
                  │
     keypoint query tokens (state+action CÓ THỂ bị nhìn, keypoint query tokens
     KHÔNG được nhìn ngược lại — self-attention mask 1 chiều)
                  │
                  ▼
       keypoint_position_decoder ──► pred_kp (full N*K, luôn full dù keypoint_n_key < N*K)
                  │
    reconstruction loss (Huber, index cố định, mask theo keypoint_valid + keypoint_n_key subsample)
```

## Quyết định thiết kế quan trọng (riêng cho `cvae`)

1. **Condition = VLM output (pooled), không phải `state`.**
   Encoder cần "biết" ít nhất bằng những gì decoder biết, để z chỉ phải mã hoá phần thực sự ngẫu
   nhiên (mode nào xảy ra) chứ không phải bù cho context yếu. Dùng `state` (chỉ proprioception,
   không thấy scene) sẽ ép z phải cõng cả thông tin thị giác → z quá "nặng", decoder ỷ lại z, bỏ qua
   conditioning thật → hỏng khi inference set z=0 (posterior collapse theo hướng xấu). Cấu hình qua
   `keypoint_cvae_condition: "vlm" | "state"` (mặc định `"vlm"`).

2. **z_style nối vào keypoint query tokens, KHÔNG nối vào `sa_embs`.**
   `sa_embs` (state+action) được dùng chung để `action_decoder` ra `action_pred`. Nếu z_style (vốn
   được encoder tính từ **label thật** lúc train) lọt vào `sa_embs`, `action_pred` sẽ "nhìn trộm"
   tương lai lúc train nhưng lúc infer z=0 (không có gì) → lệch train/inference, hỏng action một
   cách âm thầm (train loss trông tốt giả tạo). Thay vào đó, z_style được cộng vào
   `keypoint_query_embedding` (cơ chế `"tokens"` mode có sẵn) — các token này bị
   `_keypoint_self_attention_mask` chặn 1 chiều: state/action **không bao giờ** attend ngược lại
   keypoint query tokens, nên `action_pred` **provably unaffected**, đúng guarantee mà `"default"`/
   `"tokens"` mode đã có.

3. **Encoder luôn thấy FULL N*K điểm, decoder loss có thể chỉ dùng `keypoint_n_key` điểm/step.**
   Tách rời 2 việc: "encoder nhìn label thật để nén z" (luôn full, không phụ thuộc subsample) và
   "decoder loss được tính trên bao nhiêu điểm mỗi step" (`keypoint_n_key`, resample ngẫu nhiên mỗi
   step qua `_sample_keypoint_indices`, mặc định `None` = dùng hết N*K — thử nghiệm đầu tiên theo kế
   hoạch). Decoder luôn output full N*K (`keypoint_position_decoder` không đổi shape theo
   `keypoint_n_key`), nên đổi `keypoint_n_key` sau này không cần đổi checkpoint. `key_indices` cũng
   là tham số optional của `_compute_keypoint_position_loss` — 3 mode kia luôn gọi không truyền
   (None = dùng hết).

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
| `keypoint_n_key` | `None` | `cvae` | `None` = dùng hết N*K; đặt số nhỏ hơn để subsample mỗi step |

`keypoint_active_loss_weight` đã bị **xoá** (không còn field này trong config) — không mode nào còn
active loss để weight nữa.

## Loss & log keys

- `keypoint_loss` — reconstruction Huber, trọng số `keypoint_loss_weight`, **mọi mode** đều report
  key này (kể cả `share_dim`, tính trực tiếp trên velocity thay vì qua decoder riêng).
- `keypoint_kl_loss` — chỉ `cvae`, trọng số `keypoint_kl_weight`. `trainer.py` chỉ track/log key này
  khi `keypoint_head_mode == "cvae"`.
- `keypoint_active_loss` — **đã bị xoá hoàn toàn**, không mode nào emit key này nữa.

## Data pipeline: KHÔNG cần đổi

Tái dùng nguyên schema cũ (`keypoint_target`, `keypoint_active_target`, `has_keypoint`) — chỉ đổi
**ý nghĩa** của `keypoint_active_target`: từ "active theo từng frame" (bản gốc) sang "valid mask tĩnh
theo object, lặp lại giống nhau ở mọi horizon step" (bản simple pipeline, xem
`observation.keypoint_valid` trong `keypoint_tracking_simple_pipeline.md`). Không cần sửa
`gr00t/data/dataset/lerobot_episode_loader.py` hay `processing_gr00t_n1d7.py`.

## Checkpoint migration (`gr00t/model/gr00t_n1d7/setup.py`)

Load từ checkpoint cũ (train trước khi đồng bộ hoá) sẽ tự động:
- Bỏ qua (discard, không lỗi) mọi weight `keypoint_active_decoder.*` tìm thấy trong checkpoint —
  coi như retired param, giống cách `keypoint_decoder.` (bản combined cũ hơn) đã được xử lý.
- Với `share_dim`: `action_dim` mismatch giờ lớn hơn nhiều (N*K*2 thay vì N) — splice logic
  (`action_dim_mismatched`) không đổi cách hoạt động, chỉ đổi độ rộng; pretrained action weights vẫn
  được ghép vào leading slice của tensor mới.

## Inference

`get_action_with_features`: `keypoint_pred` decode khác nhau theo mode (`share_dim` đọc trực tiếp từ
`actions[..., real_action_dim:]`, 3 mode kia gọi `_decode_keypoint_positions`). `cvae`: z_style mặc
định = `zeros`. Debug/eval có thể truyền
`options={"return_keypoints": True, "keypoint_style_sample": True}` để sample z_style ~ N(0,I) thay
vì 0 (xem đa dạng future dự đoán) — không ảnh hưởng `action_pred` dù chọn cách nào.
`keypoint_active_pred` trả về hằng số 1 (mọi điểm luôn "on") ở **mọi mode**, chỉ để code overlay
visualization cũ (`keypoint_viz.py`, `trainer.py`) chạy không cần sửa.

## Tham chiếu code

- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`:
  `_compute_keypoint_position_loss` (dùng chung mọi mode, tham số `key_indices` optional),
  `_share_dim_position_targets`, `_init_keypoint_cvae_modules`, `_keypoint_condition_token`,
  `_sample_keypoint_indices`, `_encode_keypoint_style`, `_sample_keypoint_style`,
  `_compute_keypoint_kl_loss`, `_append_keypoint_queries` (tham số `z_style`).
- `gr00t/configs/model/gr00t_n1d7.py`: docstring đầy đủ trong `keypoint_head_mode`.
- `gr00t/model/gr00t_n1d7/setup.py`: `_create_model` — checkpoint migration cho các retired param.
- `gr00t/experiment/trainer.py`: log `keypoint_kl_loss` khi `keypoint_head_mode == "cvae"`.
