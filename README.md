## BÁN KẾT DATAFLOW 2026 – Dự đoán hành vi người dùng (User Behavior Prediction)

Repo này chứa toàn bộ mã nguồn và thí nghiệm cho bài toán **phân loại đa đầu ra (multi‑output classification)** dự đoán **6 thuộc tính hành vi độc lập** của người dùng từ **chuỗi hành động giao dịch trong quá khứ**, phục vụ tối ưu vận hành kho bãi và kế hoạch nhập hàng.

### 1. Mô tả bài toán
- **Đầu vào (`X_train`, `X_val`, `X_test`)**:  
  - Mỗi dòng là **một khách hàng / một phiên giao dịch**.  
  - Một chuỗi hành động (event) đã được **mã hóa thành các ID số**, độ dài chuỗi **không cố định** (variable‑length sequence).  
- **Đầu ra (`Y_train`, `Y_val`)**:  
  - 6 cột mục tiêu: `attr_1` … `attr_6` (kiểu `uint16`).  
  - Mỗi thuộc tính phản ánh một khía cạnh hành vi mua sắm khác nhau, **độc lập** về mặt business.
- **Yêu cầu nộp bài**:  
  - File `ten_doi_thi.csv` với các cột: `id, attr_1, ..., attr_6` cho toàn bộ ID trong `X_test.csv`.  
  - Chỉ số xếp hạng trên Kaggle: **Exact‑Match Accuracy** – một mẫu được tính đúng khi **cả 6 giá trị** đều dự đoán chính xác.

Ràng buộc: **không sử dụng LLM > 0.5B tham số**, tuân thủ quy tắc tách **Train / Validation** trong giai đoạn EDA và đánh giá mô hình.

### 2. Cấu trúc thư mục
- **`data/`**:  
  - `X_train.csv`, `Y_train.csv`  
  - `X_val.csv`, `Y_val.csv`  
  - `X_test.csv`
- **`src/`**  
  - `best_tranformer.py`: Pipeline Transformer + K‑Fold + Pseudo‑label + Temperature Scaling → **mô hình cuối cùng tốt nhất**.  
  - `best_GRU.py`: BiGRU + attention + lookup heuristic, dùng cho baseline mạnh & phân tích thêm.  
  - `CNNv1.ipynb`: Thử nghiệm CNN trên chuỗi hành động.  
  - `lgbm_01_feature extract.ipynb`, `lgbm_02_feature_engineering.ipynb`, `lgbm_03_lgmk_model.ipynb`: Pipeline LightGBM trên feature chuỗi.  
- **`submissions/`**  
  - `submission_B_transformer.csv`: Submission tốt nhất từ mô hình Transformer.  
  - `CNN submission.csv`: Submission từ mô hình CNN.  
- `new_eda.ipynb`: EDA, profiling hành vi và phân tích bất thường.
- **pipeline_data/**: Thư mục chứa các artifact trung gian của pipeline dữ liệu được tạo sau bước EDA và preprocessing. Các file trong thư mục này giúp chuẩn hóa dữ liệu đầu vào cho quá trình huấn luyện mô hình và đảm bảo pipeline có thể tái tạo nhất quán.
  - sequences.pkl: Lưu trữ chuỗi hành động của người dùng sau khi xử lý, bao gồm ba tập train, val, test. Mỗi phần tử là một danh sách token integer biểu diễn sequence hành vi.
  - meta.pkl: Metadata của dataset, bao gồm danh sách cột metadata (meta_cols), cột sequence (seq_cols), chiều dài chuỗi p95 (max_len_p95) và kích thước tập train (N_train). Thông tin này được sử dụng để cấu hình input của mô hình.
  - encoders.pkl: Các LabelEncoder đã được fit trên tập train cho từng target attribute (attr_1 → attr_6). File này cho phép chuyển đổi nhãn giữa dạng số và dạng gốc trong quá trình training và inference.
  - token_stats.pkl: Thống kê token trong chuỗi hành động, bao gồm tần suất token (counter) và phân loại token theo mức độ xuất hiện (very_common, common, rare). Các thống kê này phục vụ phân tích phân phối hành vi và thiết kế embedding.
  - Y_train_enc.csv, Y_val_enc.csv: Nhãn mục tiêu sau khi được encode thành số nguyên để dùng trực tiếp trong quá trình huấn luyện mô hình.
  - Y_train_orig.csv, Y_val_orig.csv: Nhãn gốc trước khi encode, được lưu lại để tiện kiểm tra và chuyển đổi kết quả dự đoán về dạng label ban đầu.

### 3. Các mô hình và kết luận (theo rubric)
- **EDA & Tiền xử lý (35%)**
  - Phân tích phân bố độ dài chuỗi, tần suất action, entropy, hành vi đầu/cuối phiên.  
  - Xây dựng **feature thống kê + positional** (first/last/…; bigram, 2‑way, 3‑way interaction) và phát hiện các pattern hành vi lặp lại / bất thường.  
  - Tuân thủ tách **Train / Val** trong giai đoạn EDA, chỉ gộp Train+Val khi retrain cuối cùng cho submission.

- **Các kiến trúc đã thử (≥ 3 mô hình)**  
  - **BiGRU + Attention (`best_GRU.py`)**:  
    - Học representation chuỗi bằng GRU hai chiều, kết hợp attention pooling và nhánh feature phụ.  
    - Cho kết quả tốt, làm baseline mạnh và dễ giải thích (per‑attr head, per‑feature importance).  
  - **CNN trên chuỗi (`CNNv1.ipynb`)**:  
    - Convolution trên chuỗi action, phù hợp các pattern cục bộ nhưng khó nắm bắt dài hạn và phụ thuộc mạnh vào chiều dài padding.  
  - **LightGBM trên feature chuỗi (`lgbm_*.ipynb`)**:  
    - Trích xuất nhiều feature thống kê / positional, sau đó dùng LightGBM multi‑output (hoặc từng attr).  
    - Mạnh ở tính giải thích nhưng hạn chế ở modeling thứ tự chính xác của chuỗi.  
  - **Transformer đa đầu ra + chain‑aware + pseudo‑label (`best_tranformer.py`) – MÔ HÌNH CUỐI CÙNG TỐT NHẤT**:  
    - Encoder Transformer trên toàn bộ chuỗi với positional encoding, kết hợp **auxiliary features** phong phú.  
    - **Per‑attribute attention**: mỗi thuộc tính có vector truy vấn riêng để "nhìn" vào các vị trí quan trọng trong chuỗi.  
    - **Chain‑aware heads**: một số thuộc tính (ví dụ `attr_4`, `attr_5`) dùng embedding từ dự đoán của attr khác (chain map) → khai thác phụ thuộc mềm giữa các hành vi.  
    - **K‑Fold CV + Ensemble**: 5 folds × 2 seeds → 10 models, đảm bảo tổng số tham số vẫn < 0.5B nhưng tăng độ ổn định.  
    - **Pseudo‑labeling**: dùng các dự đoán rất tự tin trên `X_test` (> 0.999) làm nhãn giả, retrain vòng 2 để mô hình thích nghi dữ liệu test.  
    - **Temperature scaling** trên logit ensemble để cân bằng độ tự tin các mô hình con.  

  → **Transformer là mô hình tốt nhất** vì:  
  - Khả năng **học trực tiếp quy luật chuỗi hành động** (order‑sensitive) tốt hơn CNN/LightGBM.  
  - Ensemble + pseudo‑label giúp **tăng Exact‑Match Accuracy trên Test** ổn định hơn GRU đơn lẻ.  
  - Per‑attr attention và chain‑aware head giúp **giải thích được** vị trí/hành vi nào ảnh hưởng tới từng thuộc tính.

### 4. Cách chạy mã & tái lập kết quả

#### 4.1. Cài đặt môi trường
Tối thiểu Python 3.9+. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

#### 4.2. Chuẩn bị dữ liệu
- Tải các file `X_train.csv`, `Y_train.csv`, `X_val.csv`, `Y_val.csv`, `X_test.csv` theo format BTC.  
- Đặt toàn bộ vào thư mục `data/` ở root repo (cùng cấp với `src/`, `requirements.txt`).

#### 4.3. Huấn luyện & tạo submission với Transformer (mô hình tốt nhất)

```bash
cd path/to/User-Behavior-Analysis
python src/best_tranformer.py
```

Script sẽ:
- Load dữ liệu Train + Val + Test.  
- Xây dựng vocabulary, encode & pad chuỗi, sinh auxiliary features.  
- Chạy **5‑fold CV × 2 seeds**, lưu lại weight tốt nhất.  
- Ensemble, hiệu chỉnh nhiệt độ, sinh `submission_A.csv` và sau pseudo‑label sinh `submission_B.csv`.  
- File tốt nhất để nộp: **`submission_B.csv`** (đã được copy vào `submissions/submission_B_transformer.csv`).

#### 4.4. Chạy baseline GRU

```bash
python src/best_GRU.py
```

Script sẽ:
- Huấn luyện BiGRU + attention trên Train, đánh giá trên Val.  
- Áp dụng **lookup override** trên các pattern positional chắc chắn.  
- Retrain trên Train+Val và xuất `submission.csv` trong thư mục `data/`.

### 5. Ghi chú về đánh giá & generalization
- **Exact‑Match Accuracy**: mọi thí nghiệm đều log cả exact‑match tổng và accuracy per‑attr (xem log in code, đặc biệt trong `best_GRU.py` và `best_tranformer.py`).  
- **Validation‑first**: mọi so sánh mô hình (GRU, CNN, LightGBM, Transformer) đều được thực hiện trên tập **Validation** tách rời để tránh leakage.  
- **Tổng quát hóa**:  
  - Mô hình Transformer không chỉ "học vẹt" các pattern đơn lẻ mà còn kết hợp **ngữ cảnh toàn chuỗi**, vị trí và thống kê, giúp áp dụng được cho các phân bố hành vi mới.  
  - Pseudo‑labeling chỉ sử dụng các mẫu có độ tự tin cực cao và vẫn giữ kênh kiểm soát bằng kết quả trên Validation ban đầu.


