<h4 align="center">Пословное распознавание русского почерка</h4>

### Таблица параметров экспериментов

| Параметр        | Эксперимент 1       | Эксперимент 2       | Эксперимент 3       | Эксперимент 4       | Эксперимент 5       |
|------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| model\*          | crnn_vgg16_bn       | crnn_vgg16_bn       | crnn_vgg16_bn       | crnn_vgg16_bn       | crnn_vgg16_bn       |
| batch_size       | 16                  | 16                  | 16                  | 16                  | 16                  |
| train_path       | train_one_word      | train_path          | train_one_word      | train_path          | new_train_one_word  |
| val_path         | val_one_word        | val_path            | val_one_word        | val_path            | new_val_one_word    |
| epochs           | 5                   | 5                   | 15                  | 17                  | 20                  |
| early-stop-epochs| 5                   | 4                   | 10                  | 15                  | 17                  |
| vocab            | russian             | russian             | russian             | russian             | russian             |
| device           | 0                   | 0                   | 0                   | 0                   | 0                   |
| max-chars        | 31                  | 31                  | 31                  | 31                  | 31                  |
| output_dir       | ./models            | ./models            | ./models            | ./models            | ./models            |
| flags\*          | early-stop, wb, amp | early-stop, wb, amp | early-stop, wb, amp | early-stop, wb, amp | early-stop, wb, amp |

запускаемый файл во время обучения: python references/recognition/train_pytorch.py