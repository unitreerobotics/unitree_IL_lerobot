# G1 Dex3 Toasted Bread: Dataset, ACT Training, and Robot Output

Tài liệu này mô tả pipeline đang được dùng trong repo:

```text
unitreerobotics/G1_Dex3_ToastedBread_Dataset
    -> LeRobotDataset
    -> batch ảnh + trạng thái + action chunk
    -> ACT Policy
    -> loss và cập nhật trọng số
    -> checkpoint
    -> inference
    -> lệnh khớp tay G1 và bàn tay Dex3
```

Mục tiêu là giải thích chính xác dữ liệu được tổ chức ra sao, model nhận gì, học gì và output cuối cùng được đưa tới robot như thế nào.

## 1. Bản chất bài toán

Đây là bài toán **imitation learning**, cụ thể là **behavior cloning** có giám sát.

Người vận hành thực hiện thao tác bằng hệ teleoperation. Hệ teleoperation chuyển chuyển động của người thành lệnh mục tiêu cho robot. Trong quá trình robot thực hiện, hệ thống ghi đồng bộ:

- Hình ảnh từ các camera.
- Trạng thái khớp thực tế của robot.
- Action mà hệ teleoperation gửi cho robot.

Model học ánh xạ:

```text
quan sát hiện tại -> chuỗi action mà người vận hành sẽ thực hiện tiếp theo
```

Viết ngắn gọn:

```text
policy(images[t], state[t]) -> action[t:t+100]
```

Đây không phải reinforcement learning trong lần train hiện tại:

- Không có reward.
- Không có online exploration.
- Không có môi trường simulation tham gia vào training.
- Model chỉ bắt chước các demonstration đã có trong dataset cố định.

## 2. Task trong dataset

Task duy nhất có tên:

```text
toasted bread
```

Theo mô tả dataset:

1. Lấy bánh mì từ khay và đặt vào máy nướng.
2. Sau khi nướng, đưa bánh cho con người.

Mỗi demonstration dài khoảng 20 đến 40 giây và được ghi ở 30 Hz.

Dataset khuyến nghị bố trí máy nướng, khay và camera gần giống cảnh trong dữ liệu. Behavior cloning thường giảm chất lượng rõ rệt khi môi trường thật khác nhiều so với dữ liệu train.

## 3. Dữ liệu được tạo ra như thế nào

Pipeline thu thập có thể hiểu như sau:

```text
Chuyển động của người vận hành
    -> thiết bị teleoperation / hand tracking
    -> retargeting từ pose người sang pose robot
    -> action mục tiêu cho khớp robot
    -> robot thực hiện action
    -> camera và encoder khớp đo trạng thái thực tế
    -> ghi images[t], state[t], action[t] cùng timestamp
```

Điểm quan trọng:

- Dataset LeRobot không cần chứa dữ liệu skeleton hoặc motion capture thô.
- Dataset hiện tại chứa action đã được hệ teleoperation và retargeting chuyển thành không gian khớp robot.
- Từ các file LeRobot hiện tại không thể kết luận đầy đủ loại kính hoặc cảm biến motion capture cụ thể đã được dùng.
- Tài liệu dataset tham chiếu hệ AVP Teleoperation của Unitree.

### State khác action

Tại một thời điểm `t`:

```text
state[t]  = vị trí khớp robot thực tế đang đo được
action[t] = vị trí khớp mục tiêu mà controller được yêu cầu đi tới
```

Hai vector cùng có 28 chiều nhưng không nhất thiết bằng nhau. Sai khác xuất hiện do:

- Độ trễ điều khiển.
- Quán tính cơ khí.
- Giới hạn tốc độ và gia tốc.
- Sai số bám của controller.
- Tiếp xúc với vật thể.

Vì vậy model học ý định điều khiển từ demonstration, không chỉ sao chép lại state hiện tại.

## 4. Quy mô và cấu trúc dataset

Dataset cache trên máy:

```text
~/.cache/huggingface/lerobot/
└── unitreerobotics/
    └── G1_Dex3_ToastedBread_Dataset/
```

Thông số thực tế trong `meta/info.json`:

| Thuộc tính | Giá trị |
|---|---:|
| LeRobot dataset version | v3.0 |
| Robot type | Unitree_G1 |
| Episodes | 418 |
| Frames | 352,022 |
| Tasks | 1 |
| Frequency | 30 Hz |
| Split | toàn bộ episode `0:418` thuộc train |
| Dung lượng local | khoảng 14 GB |

### Cấu trúc file

```text
G1_Dex3_ToastedBread_Dataset/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet
│       └── file-001.parquet
├── videos/
│   ├── observation.images.cam_left_high/
│   │   └── chunk-000/*.mp4
│   ├── observation.images.cam_right_high/
│   │   └── chunk-000/*.mp4
│   ├── observation.images.cam_left_wrist/
│   │   └── chunk-000/*.mp4
│   └── observation.images.cam_right_wrist/
│       └── chunk-000/*.mp4
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    └── episodes/
        └── chunk-000/file-000.parquet
```

Vai trò của từng nhóm:

- `data/*.parquet`: state, action, timestamp và các index của từng frame.
- `videos/**/*.mp4`: ảnh camera được nén thành video AV1.
- `meta/info.json`: schema, shape, FPS, đường dẫn và tổng số episode/frame.
- `meta/stats.json`: thống kê dùng để normalize/unnormalize.
- `meta/tasks.parquet`: ánh xạ `task_index` tới nội dung task.
- `meta/episodes/*.parquet`: vị trí bắt đầu/kết thúc và metadata của từng episode.

Các MP4 chỉ là cách đóng gói và nén ảnh. Một file MP4 vật lý có thể chứa nhiều episode. Đơn vị logic của dataset vẫn là **episode và frame**, không phải file video.

## 5. Đơn vị episode và frame

Một episode là một lần hoàn chỉnh hoặc một lần thử thực hiện task. Một frame là một mẫu ở một thời điểm trong episode:

```text
frame[t]
├── observation.images.cam_left_high
├── observation.images.cam_right_high
├── observation.images.cam_left_wrist
├── observation.images.cam_right_wrist
├── observation.state
├── action
├── timestamp
├── frame_index
├── episode_index
├── index
└── task_index
```

Các trường index có ý nghĩa:

- `timestamp`: thời gian của frame trong episode.
- `frame_index`: vị trí frame bên trong episode.
- `episode_index`: frame thuộc demonstration nào.
- `index`: index toàn cục trong dataset.
- `task_index`: liên kết tới task `"toasted bread"`.

Khi lấy mẫu, LeRobot dùng timestamp để decode đúng frame từ bốn video camera.

## 6. Bốn camera đầu vào

Mỗi frame có bốn ảnh RGB:

| Key | Ý nghĩa | Shape |
|---|---|---|
| `observation.images.cam_left_high` | camera cao bên trái | `[3, 480, 640]` |
| `observation.images.cam_right_high` | camera cao bên phải | `[3, 480, 640]` |
| `observation.images.cam_left_wrist` | camera cổ tay trái | `[3, 480, 640]` |
| `observation.images.cam_right_wrist` | camera cổ tay phải | `[3, 480, 640]` |

Video có:

- FPS: 30.
- Codec: AV1.
- Pixel format: YUV420P.
- Không có audio.

Khi đưa vào PyTorch, ảnh có dạng channel-first:

```text
[C, H, W] = [3, 480, 640]
```

Trong batch size 8, mỗi camera có shape:

```text
[B, C, H, W] = [8, 3, 480, 640]
```

## 7. State và action 28 chiều

`observation.state` và `action` dùng cùng thứ tự 28 khớp:

| Index | Nhóm | Khớp |
|---:|---|---|
| 0 | Tay trái | `kLeftShoulderPitch` |
| 1 | Tay trái | `kLeftShoulderRoll` |
| 2 | Tay trái | `kLeftShoulderYaw` |
| 3 | Tay trái | `kLeftElbow` |
| 4 | Tay trái | `kLeftWristRoll` |
| 5 | Tay trái | `kLeftWristPitch` |
| 6 | Tay trái | `kLeftWristYaw` |
| 7 | Tay phải | `kRightShoulderPitch` |
| 8 | Tay phải | `kRightShoulderRoll` |
| 9 | Tay phải | `kRightShoulderYaw` |
| 10 | Tay phải | `kRightElbow` |
| 11 | Tay phải | `kRightWristRoll` |
| 12 | Tay phải | `kRightWristPitch` |
| 13 | Tay phải | `kRightWristYaw` |
| 14 | Dex3 trái | `kLeftHandThumb0` |
| 15 | Dex3 trái | `kLeftHandThumb1` |
| 16 | Dex3 trái | `kLeftHandThumb2` |
| 17 | Dex3 trái | `kLeftHandMiddle0` |
| 18 | Dex3 trái | `kLeftHandMiddle1` |
| 19 | Dex3 trái | `kLeftHandIndex0` |
| 20 | Dex3 trái | `kLeftHandIndex1` |
| 21 | Dex3 phải | `kRightHandThumb0` |
| 22 | Dex3 phải | `kRightHandThumb1` |
| 23 | Dex3 phải | `kRightHandThumb2` |
| 24 | Dex3 phải | `kRightHandIndex0` |
| 25 | Dex3 phải | `kRightHandIndex1` |
| 26 | Dex3 phải | `kRightHandMiddle0` |
| 27 | Dex3 phải | `kRightHandMiddle1` |

Có thể nhóm thành:

```text
0:7    = 7 khớp tay trái
7:14   = 7 khớp tay phải
14:21  = 7 motor Dex3 trái
21:28  = 7 motor Dex3 phải
```

### Không phải toàn bộ 29 DOF của humanoid

Tên `G1_29` trong controller nói tới biến thể phần cứng G1 29 DOF. Tuy nhiên policy này chỉ output 28 giá trị cho:

- 14 DOF của hai cánh tay.
- 14 motor của hai bàn tay Dex3.

Policy không trực tiếp dự đoán:

- Chân.
- Thân dưới.
- Cân bằng toàn thân.
- Toàn bộ 29 khớp cơ thể G1.

Robot cần một motion mode hoặc controller nền giữ thăng bằng trong khi policy điều khiển phần thân trên.

## 8. Từ frame thành một training sample

ACT được cấu hình:

```text
n_obs_steps   = 1
chunk_size    = 100
n_action_steps = 100
```

Khi dataloader chọn một frame toàn cục `t`, sample logic là:

```text
Input:
    image_left_high[t]       [3, 480, 640]
    image_right_high[t]      [3, 480, 640]
    image_left_wrist[t]      [3, 480, 640]
    image_right_wrist[t]     [3, 480, 640]
    state[t]                 [28]

Target:
    action[t:t+100]          [100, 28]
    action_is_pad            [100]
```

Vì FPS bằng 30:

```text
100 action / 30 Hz = khoảng 3.33 giây tương lai
```

Model nhìn một observation hiện tại và học dự đoán khoảng 3.33 giây chuyển động tiếp theo.

### Không đi xuyên qua biên episode

Nếu `t` gần cuối episode và không còn đủ 100 action:

- LeRobot kẹp index còn thiếu vào frame cuối của episode.
- `action_is_pad[i] = true` đánh dấu vị trí không có dữ liệu tương lai thật.
- Loss bỏ qua các vị trí padding.
- Chuỗi không bao giờ lấy action từ episode tiếp theo.

Ví dụ episode chỉ còn 30 frame:

```text
30 action đầu: dữ liệu thật
70 action sau: padding, không đóng góp vào loss
```

## 9. Cách chia và shuffle dữ liệu

Dataset hiện tại khai báo:

```text
train = episode 0:418
```

Nghĩa là:

- Cả 418 episode đều đang dùng để train.
- Chưa có validation split riêng.
- Dataloader shuffle các frame/sample trong tập train.
- Batch không cần chứa các frame liên tiếp.

Batch size hiện tại là 8. Một batch có shape chính:

```text
state:          [8, 28]
4 x image:      mỗi tensor [8, 3, 480, 640]
action:         [8, 100, 28]
action_is_pad:  [8, 100]
```

Nếu tạo validation/test, cần chia theo **episode**, không chia ngẫu nhiên theo frame. Các frame liền nhau rất giống nhau; chia theo frame sẽ làm rò rỉ gần như cùng một chuyển động sang cả train và validation.

Ví dụ hợp lý:

```text
Train:      90% episode
Validation: 10% episode
```

Nếu cần đánh giá khả năng tổng quát hóa mạnh hơn, nên tách theo phiên thu thập hoặc cách bố trí vật thể, thay vì chỉ tách episode ngẫu nhiên.

## 10. Preprocessing và normalization

Trước khi gọi policy, batch đi qua `preprocessor`.

### Ảnh

Ảnh được:

1. Decode từ MP4 theo timestamp.
2. Chuyển thành tensor float.
3. Đưa về layout `[C, H, W]`.
4. Normalize theo mean/std.

Cấu hình có:

```text
dataset.use_imagenet_stats = true
```

Vì vậy thống kê ImageNet được dùng cho camera:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Image augmentation hiện đang tắt:

```text
image_transforms.enable = false
```

### State và action

State và action được normalize theo `MEAN_STD`, sử dụng thống kê trong `meta/stats.json`:

```text
x_normalized = (x - mean) / std
```

Trong training:

- `state` được normalize trước khi vào model.
- target `action` được normalize trước khi tính loss.

Trong inference:

- state và ảnh đi qua đúng preprocessor đã lưu.
- output normalized của model đi qua postprocessor.
- postprocessor unnormalize action về miền giá trị khớp thật.

Không nên chỉ load `model.safetensors` rồi bỏ qua processor. Nếu thiếu đúng mean/std, model vẫn chạy nhưng action sẽ sai scale.

## 11. Kiến trúc ACT đang train

Checkpoint hiện tại dùng:

| Thành phần | Cấu hình |
|---|---|
| Policy | ACT |
| Vision backbone | ResNet-18 pretrained ImageNet |
| Transformer hidden dimension | 512 |
| Attention heads | 8 |
| Feed-forward dimension | 3200 |
| Transformer encoder layers | 4 |
| Transformer decoder layers | 1 |
| Action queries | 100 |
| Action dimension | 28 |
| VAE | bật |
| Latent dimension | 32 |
| VAE encoder layers | 4 |
| Dropout | 0.1 |
| KL weight | 10 |

Tổng model có khoảng 52 triệu tham số.

### 11.1 Image encoder

Mỗi camera được đưa độc lập qua cùng một ResNet-18:

```text
image [B, 3, 480, 640]
    -> ResNet-18
    -> spatial feature map
    -> 1x1 convolution
    -> feature dimension 512
```

Mỗi vị trí không gian trên feature map trở thành một visual token. Token giữ thông tin về:

- Vật thể nào xuất hiện.
- Vị trí tương đối của bánh, khay, máy nướng và tay.
- Hình dạng và trạng thái thao tác nhìn thấy từ từng camera.

Model thêm 2D sinusoidal positional embedding để phân biệt vị trí của các visual token.

### 11.2 State token

Vector state:

```text
[B, 28]
```

được chiếu tuyến tính:

```text
Linear(28 -> 512)
```

để tạo một robot-state token. Token này cho model biết cấu hình khớp hiện tại, bổ sung cho thông tin hình ảnh.

### 11.3 Latent token của VAE

Khi training, VAE encoder nhận:

```text
[CLS token, state token, 100 action tokens]
```

Action `[B, 100, 28]` được chiếu từ 28 lên 512 chiều. Output của CLS token được chiếu thành:

```text
mu        [B, 32]
log_var   [B, 32]
```

Sau đó lấy mẫu:

```text
z = mu + exp(0.5 * log_var) * epsilon
epsilon ~ N(0, I)
```

Latent `z` giúp ACT biểu diễn nhiều kiểu chuyển động hợp lệ có thể xuất hiện trong demonstration.

Khi inference không có ground-truth action tương lai, code đặt:

```text
z = 0
```

Do đó inference là xác định với cùng input, nếu các nguồn khác không tạo nhiễu.

### 11.4 Transformer encoder

Input của Transformer encoder gồm:

```text
latent token
+ state token
+ visual tokens camera trái cao
+ visual tokens camera phải cao
+ visual tokens cổ tay trái
+ visual tokens cổ tay phải
```

Self-attention cho phép các token trao đổi thông tin. Ví dụ model có thể liên kết:

- Feature bánh mì từ camera cao.
- Feature ngón tay từ camera cổ tay.
- Góc khớp hiện tại.
- Vị trí mục tiêu trong máy nướng.

### 11.5 Transformer decoder

Decoder có 100 learnable action-query position embeddings, tương ứng 100 vị trí tương lai:

```text
query 0  -> action tại t
query 1  -> action tại t+1
...
query 99 -> action tại t+99
```

Decoder cross-attend tới output của encoder. Action head:

```text
Linear(512 -> 28)
```

tạo output:

```text
actions_hat [B, 100, 28]
```

## 12. Query, Key và Value trong Transformer

Query, Key và Value không được gán cứng rằng:

```text
key = góc khớp
value = image feature
```

Với token feature `X`, model học ba phép chiếu:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Attention:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Trực giác:

- Query: token hiện tại đang cần tìm thông tin gì.
- Key: token này có đặc trưng gì để query so khớp.
- Value: nội dung sẽ được tổng hợp nếu query và key phù hợp.

Trong decoder cross-attention của ACT:

- Query đến từ 100 action-query của decoder.
- Key và Value đến từ output encoder chứa ảnh, state và latent.
- Mỗi action-query học cách lấy thông tin cần thiết để dự đoán một bước tương lai.

Ví dụ action-query ở vị trí 20 có thể chú ý mạnh tới:

- Vị trí bánh trong camera.
- Khoảng cách bàn tay tới bánh.
- Góc cổ tay hiện tại.

Nhưng đây là hành vi học được. Không có quy tắc cố định rằng một loại sensor luôn chỉ tạo Key hoặc chỉ tạo Value.

## 13. Forward pass đầy đủ

Với batch size 8:

```text
4 camera:
    4 x [8, 3, 480, 640]

state:
    [8, 28]

ground-truth action chunk:
    [8, 100, 28]

                         TRAINING
                            |
             +--------------+--------------+
             |                             |
       Image + state                  State + actions
             |                             |
        ResNet + proj                 VAE encoder
             |                             |
       visual/state tokens            mu, log_var
             |                             |
             +---------- latent z ----------+
                            |
                  Transformer encoder
                            |
                    encoded memory
                            |
            100 decoder action queries
                            |
                  Transformer decoder
                            |
                    Linear(512, 28)
                            |
              predicted action [8,100,28]
```

## 14. Loss function

ACT dùng hai thành phần loss.

### 14.1 Reconstruction loss

Model so sánh action dự đoán và action demonstration bằng L1:

```text
L_action = mean(|action_target - action_pred|)
```

Các vị trí có `action_is_pad = true` bị mask khỏi loss.

### 14.2 KL divergence

VAE latent được ép gần phân phối chuẩn:

```text
L_KL = D_KL(q(z | state, actions) || N(0, I))
```

Loss tổng:

```text
L_total = L_action + 10 * L_KL
```

Mục đích:

- L1 giúp dự đoán action giống demonstration.
- KL giúp không gian latent có cấu trúc và cho phép inference dùng prior chuẩn.

## 15. Một training step diễn ra thế nào

Mỗi step:

1. Dataloader lấy 8 sample ngẫu nhiên.
2. Decode 4 ảnh cho mỗi sample.
3. Lấy state hiện tại và 100 action tương lai cùng episode.
4. Preprocessor normalize dữ liệu.
5. ACT forward và tạo `[8, 100, 28]`.
6. Tính L1 loss có mask và KL loss.
7. Backpropagation.
8. Clip gradient norm tối đa 10.
9. AdamW cập nhật trọng số.

Cấu hình optimizer:

```text
optimizer     = AdamW
learning rate = 1e-5
weight decay  = 1e-4
betas         = [0.9, 0.999]
epsilon       = 1e-8
grad clip     = 10
```

`grad_norm` được log có thể lớn hơn 10 vì giá trị trả về thường là norm trước khi clipping. Gradient dùng để update đã được clip.

## 16. Step, epoch và mức phủ dataset

Training đặt:

```text
steps = 100,000
batch_size = 8
```

Tổng số sample frame được draw, có lặp:

```text
100,000 * 8 = 800,000 samples
```

Dataset có 352,022 frame, nên mức tương đương:

```text
800,000 / 352,022 ~= 2.27 lượt qua dataset
```

Đây chỉ là epoch-equivalent vì dataloader chạy vòng lặp vô hạn và shuffle sample.

Ở checkpoint 20,000:

```text
20,000 * 8 / 352,022 ~= 0.45 lượt tương đương
```

Do đó checkpoint 20K là model thật đã học, nhưng vẫn là checkpoint trung gian.

## 17. Checkpoint và output của training

Output directory:

```text
unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/
```

Checkpoint hiện có:

```text
checkpoints/020000/
├── pretrained_model/
│   ├── model.safetensors
│   ├── config.json
│   ├── train_config.json
│   ├── policy_preprocessor.json
│   ├── policy_preprocessor_step_3_normalizer_processor.safetensors
│   ├── policy_postprocessor.json
│   └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── training_state/
    ├── training_step.json
    ├── optimizer_state.safetensors
    ├── optimizer_param_groups.json
    └── rng_state.safetensors
```

Ý nghĩa:

- `model.safetensors`: trọng số policy.
- `config.json`: kiến trúc và feature schema.
- `train_config.json`: toàn bộ cấu hình lần train.
- preprocessor/postprocessor: normalization cần cho inference.
- optimizer state: cần để resume đúng momentum của AdamW.
- RNG state: khôi phục trạng thái random.

Checkpoint được lưu mỗi 20,000 step. Symlink:

```text
checkpoints/last
```

trỏ tới checkpoint mới nhất.

Training hiện tại không tạo video đánh giá hoặc success rate vì:

```text
env = null
```

`eval_freq=20000` không tự tạo real-robot evaluation khi không có environment.

## 18. Inference của ACT

Ở thời điểm chạy robot:

```text
4 ảnh camera hiện tại + state 28 chiều
    -> preprocessor
    -> ACT
    -> action chunk [1, 100, 28]
    -> action queue
    -> lấy từng action [28]
    -> postprocessor
    -> controller robot
```

Với cấu hình hiện tại:

```text
n_action_steps = 100
temporal_ensemble_coeff = null
```

`select_action()` hoạt động như sau:

1. Khi queue rỗng, gọi model một lần để sinh 100 action.
2. Đưa 100 action vào queue.
3. Mỗi lần gọi tiếp theo lấy ra một action.
4. Sau khi dùng hết 100 action mới quan sát lại và gọi model lần nữa.

Điều này là open-loop trong khoảng 3.33 giây. Với robot thật, sai số có thể tích lũy trong 100 bước. Một cấu hình thận trọng hơn thường dùng `n_action_steps` nhỏ hơn `chunk_size`, ví dụ 10 đến 30, để quan sát lại thường xuyên hơn.

Lưu ý: thay đổi `n_action_steps` không nhất thiết phải train lại vì model vẫn dự đoán chunk 100; nó chỉ quyết định bao nhiêu action được thực thi trước khi replan.

## 19. Từ output model tới robot G1

Sau postprocessing, mỗi action là:

```text
action_np [28]
```

Code chia:

```python
arm_action = action_np[:14]
left_hand_action = action_np[14:21]
right_hand_action = action_np[21:28]
```

### Hai cánh tay

`arm_action` gồm 14 target joint positions:

```text
7 tay trái + 7 tay phải
```

Controller gọi:

```text
tau = arm_ik.solve_tau(arm_action)
arm_ctrl.ctrl_dual_arm(arm_action, tau)
```

Tức là policy cung cấp joint target, còn lớp controller/IK tạo phần torque hỗ trợ và gửi lệnh tới phần cứng.

### Hai bàn tay Dex3

Mỗi bàn tay có 7 target:

```text
left  = action[14:21]
right = action[21:28]
```

Target được ghi vào shared memory. Process `Dex3_1_Controller` đọc target và publish qua DDS:

```text
rt/dex3/left/cmd
rt/dex3/right/cmd
```

State bàn tay được subscribe từ:

```text
rt/dex3/left/state
rt/dex3/right/state
```

Dex3 controller hiện đặt:

```text
kp = 1.5
kd = 0.2
```

và gửi target position `q` cho từng motor.

## 20. Closed-loop thực tế

Toàn hệ thống là closed-loop ở cấp các lần gọi policy:

```text
Robot thực hiện action
    -> camera mới + state mới
    -> policy dự đoán chunk mới
    -> robot thực hiện tiếp
```

Nhưng bên trong một chunk, với `n_action_steps=100`, policy chạy open-loop 100 bước trước khi replan.

Cần phân biệt hai controller:

- Policy controller: quyết định target joint position dựa trên ảnh và state.
- Low-level robot controller: bám target, giữ ổn định và giao tiếp motor ở tần số cao hơn.

Policy không nên trực tiếp thay thế toàn bộ low-level safety controller của robot.

## 21. Điều model đã học và chưa học

Model có thể học:

- Liên hệ giữa hình ảnh vật thể và chuyển động tay.
- Phối hợp hai tay.
- Cách thay đổi góc ngón Dex3 theo giai đoạn thao tác.
- Chuỗi chuyển động điển hình trong demonstration.

Model không tự động có:

- Hiểu biết vật lý ngoài phân phối dữ liệu.
- Cơ chế tránh va chạm bảo đảm.
- Bảo đảm giới hạn khớp, tốc độ hoặc torque.
- Khả năng tự hồi phục từ mọi sai lệch.
- Khả năng điều khiển chân và cân bằng toàn thân.
- Success metric chỉ từ training loss.

Loss thấp hơn chỉ nói model bắt chước action trong dataset tốt hơn. Thành công thật phải được đo bằng validation offline, simulation hoặc real-robot rollout.

## 22. Đánh giá model đúng cách

### Offline

Nên tạo validation split theo episode và đo:

- L1/MAE trên action normalized.
- MAE theo đơn vị khớp sau unnormalize.
- MAE riêng cho arm và hand.
- Sai số theo horizon: bước gần `t` so với bước xa `t+99`.
- Vẽ predicted action và ground-truth action theo thời gian.

### Replay hoặc simulation

Kiểm tra:

- Action có liên tục không.
- Có vượt giới hạn khớp không.
- Tốc độ và gia tốc có quá lớn không.
- Tay trái/phải có đúng mapping không.
- Camera order có đúng như lúc train không.

### Robot thật

Thứ tự kiểm tra nên là:

1. Load checkpoint và processors.
2. Chạy inference nhưng chưa gửi lệnh motor.
3. Log action, state và ảnh.
4. Kiểm tra range từng joint.
5. Clamp position, velocity và acceleration.
6. Khởi tạo robot về pose đầu episode.
7. Chạy tốc độ thấp, có emergency stop.
8. Sau khi ổn định mới tăng tần số và phạm vi thao tác.

## 23. Các rủi ro distribution shift

Policy có thể thất bại nếu khác dữ liệu train về:

- Vị trí và góc máy nướng.
- Vị trí khay hoặc bánh.
- Ánh sáng.
- Camera intrinsic/extrinsic.
- Độ phân giải, crop, channel order.
- Pose ban đầu của tay.
- Loại bánh hoặc hình dạng vật thể.
- Tần số control.
- Mapping và dấu của joint.

Đặc biệt cần giữ đúng:

```text
camera key
camera order
RGB/BGR convention
image shape
state joint order
action joint order
normalization statistics
control frequency
```

Sai joint order vẫn có thể tạo tensor đúng shape nhưng làm robot chuyển động sai hoàn toàn.

## 24. Trạng thái training hiện tại

Lần kiểm tra gần nhất:

```text
checkpoint = 020000
training step = 20,000 / 100,000
```

Checkpoint 20K đã chứa model dùng được cho inference thử nghiệm. Tuy nhiên:

- Chưa train đủ cấu hình 100K.
- Chưa có validation split.
- Chưa có real-robot success rate.
- Chưa có video eval do `env=null`.

Resume:

```bash
conda activate unitree_lerobot
cd /home/jkl0909/Code/rl/unitree_lerobot/unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_train.py \
  --config_path=outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  2>&1 | tee -a outputs/g1_dex3_toastedbread_act/training.log
```

## 25. Các file code quan trọng trong repo

| Chức năng | File |
|---|---|
| Schema G1 Dex3 và thứ tự 28 khớp | `unitree_lerobot/utils/constants.py` |
| Chuyển Unitree JSON sang LeRobot | `unitree_lerobot/utils/convert_unitree_json_to_lerobot.py` |
| Tạo dataset và action delta timestamps | `unitree_lerobot/lerobot/src/lerobot/datasets/factory.py` |
| Lấy frame, padding và decode video | `unitree_lerobot/lerobot/src/lerobot/datasets/lerobot_dataset.py` |
| Cấu hình ACT | `unitree_lerobot/lerobot/src/lerobot/policies/act/configuration_act.py` |
| Forward, loss và action queue ACT | `unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py` |
| Training loop | `unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_train.py` |
| Preprocess và gọi `select_action` | `unitree_lerobot/lerobot/src/lerobot/utils/control_utils.py` |
| Vòng lặp chạy G1 thật | `unitree_lerobot/eval_robot/eval_g1.py` |
| Tạo arm/hand controller | `unitree_lerobot/eval_robot/make_robot.py` |
| DDS controller bàn tay Dex3 | `unitree_lerobot/eval_robot/robot_control/robot_hand_unitree.py` |

## 26. Tóm tắt tensor end-to-end

```text
DATASET FRAME t
    4 images               4 x [3,480,640]
    state                  [28]
    future actions         [100,28]
    padding mask           [100]

                DataLoader batch_size=8

BATCH
    4 images               4 x [8,3,480,640]
    state                  [8,28]
    actions                [8,100,28]
    action_is_pad          [8,100]

                Normalize

ACT
    ResNet-18              image -> visual tokens
    Linear                 state 28 -> token 512
    VAE encoder            actions -> latent 32
    Transformer encoder    fuse visual/state/latent
    Transformer decoder    100 action queries
    Action head            512 -> 28

MODEL OUTPUT
    predicted actions      [8,100,28]

TRAINING
    L1 masked + 10 * KL
    backward
    AdamW update

INFERENCE
    one observation
    -> [1,100,28]
    -> action queue
    -> one [28] action at a time
    -> unnormalize

ROBOT
    [0:14]   dual-arm joint targets
    [14:21]  left Dex3 targets
    [21:28]  right Dex3 targets
```

## 27. Kết luận

Toàn bộ hệ thống học một hàm điều khiển có điều kiện:

```text
f_theta(
    bốn góc nhìn camera hiện tại,
    cấu hình 28 khớp hiện tại
)
=
100 target action tương lai cho hai tay và hai bàn tay
```

Dataset được tổ chức theo episode và frame. Video chỉ là lớp lưu trữ ảnh. ACT dùng ResNet để trích xuất feature hình ảnh, Transformer để hợp nhất ảnh, state và latent, rồi dùng 100 action queries để dự đoán action chunk. Khi deploy, output được unnormalize, chia thành 14 target cho hai cánh tay và 7 target cho mỗi bàn tay Dex3, sau đó gửi qua các controller phần cứng.

Checkpoint là kết quả học được, nhưng chất lượng vận hành không thể kết luận chỉ từ training loss. Validation theo episode, kiểm tra action offline và rollout có kiểm soát là các bước bắt buộc trước khi chạy robot thật.
