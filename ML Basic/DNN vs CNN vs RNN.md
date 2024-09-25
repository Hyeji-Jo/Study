# DNN
![image](https://github.com/user-attachments/assets/280c5d4f-4461-42ab-9fa0-f96abe5f1665)

- 여러 개의 hidden layer를 hidden layer 를 2개 이상 가진 인공신경망
- **문제점**
  - 은닉층이 깊어지면서 입력층으로 갈 수록 gradient vanishing(기울기 소실) 문제가 발생할 수 있음
  - overfittiong(과적합) 문제가 발생

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 데이터 준비 (1000개 샘플, 각 샘플당 20개의 특성)
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 2, (1000, 1)).float()  # 이진 분류 (0 또는 1)

# DNN 모델 정의
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(20, 64)  # 입력층에서 첫 번째 은닉층으로
        self.fc2 = nn.Linear(64, 32)  # 첫 번째 은닉층에서 두 번째 은닉층으로
        self.fc3 = nn.Linear(32, 1)   # 두 번째 은닉층에서 출력층으로
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 모델, 손실 함수, 최적화기 설정
model = DNN()
criterion = nn.BCELoss()  # 이진 분류를 위한 손실 함수 (Binary Cross Entropy)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터셋과 DataLoader 구성
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 학습
for epoch in range(10):  # 10번의 epoch 동안 학습
    for batch_X, batch_y in train_loader:
        # 순전파
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()  # 이전 gradient 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
    
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# 학습 결과 평가
with torch.no_grad():  # 평가 시에는 gradient 계산 불필요
    outputs = model(X_train)
    predicted = (outputs > 0.5).float()  # 0.5를 기준으로 이진 분류
    accuracy = (predicted == y_train).sum().item() / y_train.size(0)
    print(f"Accuracy: {accuracy * 100:.2f}%")
```

# CNN
![image](https://github.com/user-attachments/assets/36636f00-49da-46d2-b1f4-9505cdd0a8ec)

- 합성곱 연산을 사용해 데이터를 지역적으로 처리하면서 특징을 추출하는 구조
- CNN은 크게 3가지 주요 레이어로 구성
  - 합성곱(Convolution) 레이어: 필터(커널)를 사용해 입력 데이터에서 특징을 추출하는 역할을 합니다. 필터가 이미지의 작은 영역을 스캔하면서 엣지, 코너 등의 특성을 학습
  - 풀링(Pooling) 레이어: 데이터를 축소하여 특성 맵의 크기를 줄이고, 중요한 정보만 남깁니다. 가장 많이 사용되는 방식은 '최대 풀링'으로, 특정 영역에서 최대 값을 추출합니다.
  - 완전 연결층(Fully Connected Layer): 합성곱과 풀링을 통해 추출된 특징들을 바탕으로 최종 분류 작업을 수행합니다. 마지막 출력층에서는 분류를 위한 소프트맥스 함수나 시그모이드 함수가 적용됩니다. 
- **문제점**
  - 많은 데이터 요구: CNN은 학습에 많은 데이터를 요구합니다. 대규모 데이터셋이 없으면 과적합(overfitting)의 위험이 높아집니다. 적은 데이터셋을 사용할 경우, 일반화 능력이 떨어질 수 있습니다.
  - 계산 비용: CNN은 특히 깊은 네트워크일수록 계산량이 많아집니다. 여러 층에 걸친 합성곱 연산, 풀링, 완전 연결층 등이 있어 GPU와 같은 고성능 하드웨어가 필요하고, 학습 시간이 오래 걸릴 수 있습니다.

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 데이터 전처리 (Data Augmentation 및 Normalization)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 데이터 증강: 좌우 반전
    transforms.RandomCrop(32, padding=4),  # 데이터 증강: 랜덤 크롭
    transforms.ToTensor(),  # Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 데이터 정규화
])

# CIFAR-10 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 3채널 입력 -> 32채널 출력, 필터 크기 3x3, 패딩 1, 스트라이드 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # 32채널 입력 -> 64채널 출력, 필터 크기 3x3, 패딩 1, 스트라이드 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 풀링 레이어: 2x2 크기의 최대 풀링
        self.pool = nn.MaxPool2d(2, 2)
        
        # 완전 연결층(FC): 64채널 * 8 * 8 -> 512 노드
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        
        # 완전 연결층(FC): 512 -> 10 (CIFAR-10의 클래스 개수)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # 첫 번째 합성곱 -> ReLU -> 풀링
        x = self.pool(F.relu(self.conv1(x)))
        
        # 두 번째 합성곱 -> ReLU -> 풀링
        x = self.pool(F.relu(self.conv2(x)))
        
        # 특징 맵을 1차원 벡터로 변환 (Flatten)
        x = x.view(-1, 64 * 8 * 8)
        
        # 첫 번째 완전 연결층 -> ReLU
        x = F.relu(self.fc1(x))
        
        # 출력층
        x = self.fc2(x)
        
        return x

# 모델 생성
net = SimpleCNN()

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 손실 함수
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 학습 루프
for epoch in range(10):  # 10번의 에포크 동안 학습
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        # 옵티마이저 초기화
        optimizer.zero_grad()
        
        # 순전파 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 손실 값 출력
        running_loss += loss.item()
        if i % 100 == 99:  # 매 100 미니 배치마다 출력
            print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# 테스트 정확도 평가
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
```


# RNN
![image](https://github.com/user-attachments/assets/c98282b5-389b-4e17-99cd-61da8d82dfde)

- 시계열 또는 순차 데이터를 예측하는 딥러닝을 위한 순환 구조 신경망 아키텍처
- 입력과 출력을 시퀀스 단위로 처리하는 시퀀스 모델
- **CNN이나 DNN은 독립적인 데이터 처리**를 하지만, **RNN은** 시간의 흐름에 따라 정보가 누적되기 때문에 **시퀀스 간의 상관관계를 반영할 수 있음**
- 이전 타임스텝의 정보를 기억하고, 현재 타임스텝에서 활용할 수 있음
- 문제점
  - 장기 의존성 문제(Long-term Dependency)
  - 기울기 소실 문제 존재 

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

# 데이터 전처리
TEXT = Field(tokenize='spacy', batch_first=True)
LABEL = Field(sequential=False)

train_data, test_data = IMDB.splits(TEXT, LABEL)

# 단어 사전 구축 및 데이터 로드
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# RNN 모델 정의
class RNNModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])  # 마지막 은닉 상태를 출력으로 사용

# 모델 및 하이퍼파라미터 설정
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = len(LABEL.vocab)

model = RNNModel(input_dim, embedding_dim, hidden_dim, output_dim)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_iterator:
        optimizer.zero_grad()
        
        # 입력과 레이블
        text, labels = batch.text, batch.label
        
        # 모델 예측
        predictions = model(text)
        
        # 손실 계산 및 역전파
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_iterator):.4f}')

# 테스트 정확도 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_iterator:
        text, labels = batch.text, batch.label
        predictions = model(text)
        predicted_labels = predictions.argmax(dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
```

# Transformer
![image](https://github.com/user-attachments/assets/f39d936a-d53f-4d49-9a4f-becb05f0ce8e)

- Self-Attention 메커니즘
  - 입력 시퀀스 내의 각 단어가 다른 단어와의 관계를 학습할 수 있게 해줍니다. 각 단어는 쿼리, 키, 값으로 변환되어, 서로의 정보를 효과적으로 활용합니다.
- Multi-Head Attention
  - 여러 개의 self-attention을 병렬로 수행하여 다양한 관점에서 정보를 처리합니다. 각 헤드가 서로 다른 부분에 집중하여 더 풍부한 표현을 학습할 수 있습니다.
- Positional Encoding
  - Transformer는 입력 시퀀스의 순서를 고려하지 않기 때문에, 단어의 순서를 나타내기 위한 positional encoding을 사용합니다. 이를 통해 모델이 단어 간의 순서를 인식할 수 있게 됩니다.
- Feed-Forward Networks
  - 각 attention 층 후에 위치하는 완전 연결 층으로, 각 단어의 표현을 더욱 풍부하게 하기 위해 비선형 변환을 적용합니다.
- 인코더-디코더 구조
  - Transformer는 인코더와 디코더 두 부분으로 구성됩니다. 인코더는 입력 시퀀스를 처리하여 의미 있는 표현을 생성하고, 디코더는 이 표현을 바탕으로 출력 시퀀스를 생성합니다.
- 병렬 처리의 장점
  - RNN과 달리 순차적인 계산이 필요 없어 병렬 처리가 가능하여 학습 속도가 빠르며, 장기 의존성 문제를 효과적으로 처리합니다.


```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

# 데이터 전처리
TEXT = Field(tokenize='spacy', batch_first=True)
LABEL = Field(sequential=False)

train_data, test_data = IMDB.splits(TEXT, LABEL)

# 단어 사전 구축 및 데이터 로드
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_heads, n_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embedding_dim))  # 최대 시퀀스 길이 100
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim), 
            num_layers=n_layers
        )
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]  # Positional Encoding 추가
        transformer_output = self.transformer_blocks(embedded)
        return self.fc(transformer_output.mean(dim=1))  # 평균 풀링

# 모델 및 하이퍼파라미터 설정
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = len(LABEL.vocab)
n_heads = 8
n_layers = 4

model = TransformerModel(input_dim, embedding_dim, hidden_dim, output_dim, n_heads, n_layers)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_iterator:
        optimizer.zero_grad()
        
        # 입력과 레이블
        text, labels = batch.text, batch.label
        
        # 모델 예측
        predictions = model(text)
        
        # 손실 계산 및 역전파
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_iterator):.4f}')

# 테스트 정확도 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_iterator:
        text, labels = batch.text, batch.label
        predictions = model(text)
        predicted_labels = predictions.argmax(dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

```
