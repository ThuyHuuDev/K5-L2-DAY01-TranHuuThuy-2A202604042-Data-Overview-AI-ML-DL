# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record nói về cả bức ảnh. Bằng chứng ngay trong file: task là image_classification và record chỉ có image_width, image_height của toàn ảnh, không có bbox, mask hay tọa độ nào. Nếu là một vật thể riêng thì phải có vị trí của vật thể đó. Vì vậy “cab” không chỉ vào chiếc xe cụ thể nào, mà là lớp mà model cho là gần nhất với toàn cảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách lớp đi kèm taxonomy và checkpoint, không phải model tự nghĩ ra. Con người định nghĩa taxonomy (với ImageNet-1k là 1000 lớp) và gán nhãn dữ liệu huấn luyện.
   Model chỉ học ánh xạ ảnh sang chỉ số lớp, còn tên lớp được lưu trong checkpoint (model.names). Vì vậy model không thể dự đoán lớp nằm ngoài danh sách này (ví dụ “xe máy” hay “ùn tắc giao thông” nếu taxonomy không có), nên đầu ra luôn phải chọn trong 1000 lớp đó, kể cả khi không lớp nào thật sự khớp.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   class_id là chỉ số mà model thực sự xuất ra. Nó không mơ hồ và dùng để so khớp với nhãn khi đánh giá.
   class_name giúp con người đọc được.
   taxonomy_name cho biết hai thứ trên thuộc bảng phân loại nào. Cùng một class_id có thể mang tên khác nhau ở taxonomy khác (ID 468 chỉ là “cab” trong ImageNet-1k), và cùng một tên có thể nằm ở ID khác. Tên lớp cũng có thể bị đổi cách viết hoặc trùng nghĩa giữa các phiên bản.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline phải giúp người gán nhãn chọn đúng một lớp, nên cần nêu:
   Tiêu chí chọn: chủ thể chính (nổi bật nhất, lớn nhất, ở trung tâm) hoặc mục đích của ảnh.
   Quy tắc khi nhiều lớp ngang nhau, và quy tắc khi không lớp nào phù hợp (ví dụ “other” hoặc bỏ qua).
   Nói rõ chỉ cho một nhãn (single-label) hay cho nhiều nhãn.
   Ví dụ cho các ca khó. Ảnh traffic này là ví dụ điển hình: nhiều xe buýt, xe con, taxi cùng xuất hiện, không có chủ thể duy nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
   score chỉ là độ tin cậy của model, không phải xác nhận rằng ảnh thật sự thuộc lớp đó. Model có thể tự tin mà vẫn sai.
   Ground truth phải do con người xác nhận theo guideline, độc lập với model. Dùng đầu ra của model làm đáp án là để model chấm chính nó, nên không đo được sai số thật.
   Dữ liệu của bạn cho thấy model không chắc chắn: hạng 1 chỉ 0.510915, hạng 2 (minibus) 0.164284, hạng 3 (police_van) 0.085848. Điểm bị chia nhỏ cho nhiều lớp, và cả năm lớp đều là xe cộ, phản ánh cảnh nhiều xe khó gắn vào một nhãn.
   Score còn phụ thuộc vào checkpoint, tiền xử lý (ảnh trong biểu đồ bị letterbox, có viền đen) và hiệu chỉnh, nên không phải sự thật khách quan về ảnh.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="bus" ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Record: class_name = bus, bbox_xyxy = [93.17, 187.95, 223.01, 320.91], ảnh 640×428 pixel.
   Góc trên-trái: cách mép trái ảnh 93.17 pixel và cách mép trên ảnh 187.95 pixel (đi sang phải 93.17, đi xuống 187.95).
   Góc dưới-phải: cách mép trái 223.01 pixel và cách mép trên 320.91 pixel.
   Rộng: 223.01 − 93.17 = 129.84 pixel.
   Cao: 320.91 − 187.95 = 132.96 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Ngưỡng 0.20: 17 prediction.
   Ngưỡng 0.60: 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Giữ nhiều hộp hơn (từ 6 lên 17) thì reviewer phải xem thêm 11 hộp điểm thấp, vốn dễ sai. Cụ thể phải:
   kiểm tra từng hộp mới: có vật thật không, đúng lớp không (ví dụ potted plant, bottle);
   sửa hoặc xóa hộp sai, hộp lệch, hộp trùng nhau;
   chuyển các ca mơ hồ lên người có thẩm quyền.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Mỗi vật thể thuộc taxonomy được vẽ đúng một hộp chữ nhật sát khít quanh phần nhìn thấy của vật đó. Cạnh hộp chạm sát rìa ngoài của vật ở bốn phía, không trùm nhiều vật vào một hộp, và không bỏ vật chỉ vì nhỏ hoặc model cho điểm thấp.
   Quy tắc này áp dụng cho mọi ảnh vì chỉ dựa vào hình dạng và vị trí của vật, không phụ thuộc cảnh. Ghi kèm quy ước tọa độ: pixel, xyxy, gốc ở góc trên-trái.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Vật: person 0.61 ở mép trái bàn, chỉ thấy bàn tay hoặc cánh tay, phần còn lại nằm ngoài khung ảnh và bị bát, cốc che bớt.
   Tôi chọn viết thành quy tắc trong guideline, không escalate. Vì:
   Vật bị che hoặc cắt mép xuất hiện ở rất nhiều ảnh, cần một cách xử lý thống nhất.
   Quy tắc có thể nói rõ ràng: vẽ hộp quanh phần nhìn thấy, không đoán phần bị khuất, chỉ gán khi nhận ra được lớp từ phần nhìn thấy.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="traffic-001" ; `class_name`="bus" ; `score`=0.925745 ; số điểm polygon=`polygon_point_count`=120 ; 3–5 điểm đầu của `polygon_xy`=[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Box chỉ cho vị trí và kích thước của một hình chữ nhật, tức 4 số. Polygon (ở đây 120 điểm) cho hình dạng thật của vật: chỗ nào có vật, chỗ nào là nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id là tên riêng của từng đối tượng trong output của lab. Nhờ nó phân biệt được hai vật cùng lớp (như hai bowl, hai person) và gọi đúng một polygon khi rà soát hay ghi chú ("sửa kitchen-004").
   Không phải class ID: class ID cho biết loại (bus = 5), mọi chiếc bus đều có chung số này. instance_id chỉ từng cá thể (traffic-001 là một chiếc bus cụ thể, traffic-002 là một chiếc car với class_id 2).
   Không phải tracking ID: tracking ID nối cùng một vật qua nhiều khung hình của video. Ở đây chỉ có một ảnh tĩnh, instance_id chỉ là số thứ tự trong file này. Chạy lại với ngưỡng khác hoặc model khác thì đánh số có thể đổi, nên không dùng nó để nói “cùng một vật” giữa hai lần chạy.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Mỗi vật một polygon riêng, các điểm bám sát biên ngoài của phần nhìn thấy của vật đó: không chừa nền, không lấn sang vật khác, và đặt đủ điểm ở chỗ cong hoặc góc để hình giữ được dáng thật. Phần bị vật khác che thì cắt theo biên vật che, không đoán phần khuất.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Chỗ chọn: cái bàn gỗ (dining table 0.58) ở góc dưới trái. Các bát (bowl 0.70, 0.53, 0.74) đặt ngay trên mặt bàn, và mask của bàn chồng lên chúng.
   Chọn: dùng guideline, không escalate.
   Lý do:
   Tình huống “vật đặt trên vật khác” lặp lại ở rất nhiều ảnh, nên cần một quy tắc chung để mọi người làm giống nhau.
   Quy tắc trả lời được ngay: mask của bàn không bao gồm phần bát đặt trên nó, mỗi vật có polygon riêng và không chồng nhau.
   Chỉ escalate những ca guideline chưa nói tới, ví dụ person 0.43 ở mép trái bàn (chỉ thấy bàn tay) hay vùng tối của oven 0.49 không rõ biên. Khi đó hỏi một lần rồi ghi kết luận vào guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                         | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                     | Annotator làm gì?                                                                                                           | Reviewer xem gì?                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn cho **cả ảnh** (`class_id`, `class_name`, `taxonomy_name`), không có tọa độ. | `traffic`: `cab` chỉ 0.51, `minibus` 0.16; nhiều xe cùng lúc nên khó chọn một lớp.                                                     | Chọn **một** lớp theo tiêu chí chủ thể chính của guideline; không chép top-1 của model.                                     | Nhãn đúng taxonomy chưa; chọn chủ thể có nhất quán giữa các ảnh không.                                      |
| Phát hiện vật thể     | Một **hộp** `bbox_xyxy` (pixel) và lớp COCO-80 cho **mỗi vật**.                       | `kitchen`: số hộp đổi theo ngưỡng (17/11/6); `person 0.61` chỉ là bàn tay; `oven 0.69` trùm cả khu bếp; nồi, chảo không có lớp.        | Vẽ hộp sát phần nhìn thấy của từng vật, gán cả vật nhỏ hoặc điểm thấp; ca mơ hồ thì escalate.                               | Hộp có sát không; có bỏ sót (`dining table`, `spoon`), thừa, sai lớp hoặc gộp nhiều vật không.              |
| Instance segmentation | Một **polygon** `polygon_xy` cho **mỗi cá thể**, có `instance_id` riêng.              | `traffic-001` (bus) có 120 điểm, góc box là nền; `kitchen`: mask `dining table` chồng lên các `bowl`, biên `oven` ở vùng tối không rõ. | Vẽ polygon bám biên phần nhìn thấy, mỗi vật một polygon, không lấn sang vật khác; biên mờ thì theo guideline hoặc escalate. | Biên có bám sát không; mask có chồng nhau không; đủ điểm ở chỗ cong không; đúng lớp và `instance_id` không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Chỉ dùng các ảnh mẫu được giao cho bài lab (traffic, kitchen), chỉ xử lý trong môi trường lab, không tải lên dịch vụ bên ngoài hay chia sẻ ra ngoài, và không đưa ảnh hay file JSON chứa dữ liệu của lab vào nơi công khai.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Tôi dừng lại, không xử lý hay lưu thêm, và báo ngay cho Lab Coach (người hướng dẫn bài lab) để được hướng dẫn trước khi làm tiếp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
