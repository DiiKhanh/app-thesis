import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV
df = pd.read_csv("results/three_model/stacking.csv")

# Chuyển 'ActualOutcome' về dạng nhị phân: Poor=1, Good=0
df['ActualBinary'] = df['ActualOutcome'].map({'Poor': 1, 'Good': 0})

# Hàm đánh giá accuracy với từng threshold
def evaluate_threshold(threshold):
    predicted_binary = (df['OutcomeProbability'] >= threshold).astype(int)
    return (predicted_binary == df['ActualBinary']).mean()

# Tìm threshold tốt nhất
thresholds = np.linspace(0, 1, 1000)
accuracies = [evaluate_threshold(t) for t in thresholds]

# Lấy threshold tốt nhất
best_idx = np.argmax(accuracies)
best_threshold = thresholds[best_idx]
best_accuracy = accuracies[best_idx]

print(f"Ngưỡng tốt nhất: {best_threshold:.4f}")
print(f"Độ chính xác cao nhất: {best_accuracy:.4f}")

threshold_check = 0.6
accuracy_at_066 = evaluate_threshold(threshold_check)
print(f"Độ chính xác tại threshold = 0.6: {accuracy_at_066:.4f}")