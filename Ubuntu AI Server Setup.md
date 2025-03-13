# 1. PC 구성요소
| 부품 | 제품명 |
|------|--------------------------------------------------------------------|
| **CPU** | [AMD] 라이젠9 라파엘 7950X3D (16코어/32스레드/4.2GHz) |
| **GPU** | 3090TI *2 |
| **Mainboard** | [ASRock] X670E |
| **RAM** | DDR5 PC5-44800 CL46 코잇 [32GB] |
| **SSD** | T500 M.2 NVMe 2280 [2TB TLC] |
| **HDD** | [Western Digital] BLUE HDD 4TB WD40EZAX (3.5HDD/ SATA3/ 5400rpm/ 256MB/ CMR) |
| **Power** | [SuperFlower] SF-2000F14HP LEADEX PLATINUM (ATX/2000W) |

<img width="699" alt="image" src="https://github.com/user-attachments/assets/90e3b225-2260-4025-9560-f20ed7547296" />
<img width="553" alt="image" src="https://github.com/user-attachments/assets/3b54d47f-281f-48c3-b45c-f84e8a9ab3e4" />  
<br>
<br>
<br>


  
# 2. Ubuntu 22.04 버전 설치

## 1) Ubuntu 초기화  부팅 USB 활용

- Del 버튼으로 Bios 진입  
    - **BIOS 설정 확인**  
        - CSM(Compatibility Support Module): Disabled 유지 (UEFI 모드에서 설치)  
        - Secure Boot: Disabled로 변경 (NVIDIA 드라이버 및 CUDA 설치를 위해 필요)  
        - Boot Option #1: UEFI: SanDisk, Partition 1 (SanDisk)로 변경  

<img width="539" alt="image" src="https://github.com/user-attachments/assets/a1b8b3ec-3960-4692-a1b1-751f29c135b6" />
- USB 부팅을 통해 우분투를 새로 설치하려는 경우 = **Yes** -> **“Try or Install Ubuntu”를 선택한 후 엔터**

---    

## 2) 설치된 후 처음 부팅된 화면에서 옵션 선택  

### 1. 언어 및 키보드 설정 → English로 하는게 AI 모델링 시 문제가 별로 발생되지 않음

- 언어 선택: **English** (또는 원하는 언어)  
- 키보드 레이아웃: 기본값 **English (US)** 유지 (다른 키보드 레이아웃이 필요하면 변경)  
![image](https://github.com/user-attachments/assets/c2b94210-5105-406f-8879-8fdf17152c67)
- USB 부팅을 통해 우분투를 새로 설치하려는 경우 = **Yes**  → **“Try or Install Ubuntu”를 선택한 후 엔터  **
  
### 2. 네트워크 설정
- **DHCP 자동 설정:** 기본값을 유지하면 자동으로 IP를 가져옵니다.  
- **수동 설정이 필요한 경우:** Manual을 선택하고 IP, 서브넷 마스크, 게이트웨이 등을 입력  
- 네트워크가 필요하지 않으면 *Do not configure network”를 선택하고 넘어가도 됩니다.  

  
### 3. 설치 유형 선택 
- **“Use entire disk and set up LVM”** → 이 옵션을 선택하면 **기존 파티션이 모두 삭제**되고 LVM으로 설정됩니다.  
- 만약 **LVM을 수동 설정하고 싶다면**, “Custom storage layout”을 선택합니다.
<img width="1245" alt="image" src="https://github.com/user-attachments/assets/2d900132-e8e5-4e70-b37f-0cff14409a2a" />
<img width="1194" alt="image" src="https://github.com/user-attachments/assets/16fd5da5-208c-4ea6-a60e-80e7ce5c4c1e" />
  
- 위에는 **SSD에** 설치    
  - NVMe SSD는 HDD보다 훨씬 빠른 속도(읽기/쓰기 성능)가 나오므로 운영체제(OS), 소프트웨어, AI 모델 실행에 최적
  - OS를 HDD에 설치하면 부팅과 작업 속도가 느려짐
- 아래는 HDD에 설치

  
### 4. 디스크 파티션 설정 (LVM 구성)
- 만약 “Custom storage layout”을 선택했다면:  
    
    1.	**디스크를 선택 후 새 LVM 볼륨 그룹(VG) 생성**  
    
    •	**볼륨 그룹(VG) 이름:** vg_ubuntu  
    
    •	**사용할 디스크:** /dev/nvme0n1 (SSD)  
    
    •	**HDD /dev/sdb도 포함 가능 (LVM으로 사용자별 데이터 저장 가능)**  
    
    2.	**논리 볼륨(LV) 생성**  
    
    •	**루트(/) 파티션:** lv_root (100GB~500GB, ext4)  
    
    •	**스왑(Swap) 파티션:** lv_swap (RAM의 절반 또는 16GB)  
    
    •	**/home 파티션:** lv_home (남은 공간 활용)   
    
    •	**HDD를 /mnt/user1, /mnt/user2로 마운트 가능**  
    
    3.	**파일 시스템 선택**  
    
    •	/ 및 /home: ext4  
    
    •	Swap: swap  
    

**참고:** AI 모델 학습용 데이터가 많다면, /mnt/data 볼륨을 별도로 만드는 것도 좋습니다.  

  
### 5. 사용자 계정 및 SSH 설정

- 관리자 계정 생성 (admin 또는 사용자 지정 계정)  -> **원하는 이름 작성**
- **SSH 설치**: **“Install OpenSSH server”** 옵션을 체크 (리모트 접속 필요 시 필수)  

  
### 6. 우분투 설치 및 부팅 설정

- 모든 설정을 확인하고 **설치를 진행**  
- 설치 완료 후, 재부팅하면 새롭게 설정된 LVM 기반 우분투 서버로 진입 가능


  
---   

## 3) HDD LVM(Logical Volume Manager) 활용 (유연한 용량 조정)
### 1. HDD(4TB) 마운트 확인 및 설정
- 마운트 확인
  - /dev/sda 또는 /dev/sdb로 표시될 가능성이 높음
  - /boot/efi로 마운트된 512MB의 작은 파티션만 보이며, 나머지 3.6TB는 마운트되지 않은 상태
```
lsblk

# 결과
sda         8:0    0   3.6T  0 disk 
└─sda1      8:1    0   512M  0 part /boot/efi
```  
<br>

- 기존 파티션 확인
  - 출력 결과에서 sda1(512MB) 외에 추가적인 파티션이 없다면, 새롭게 파티션을 생성할 수 있음  
```
sudo fdisk -l /dev/sda
```  
<br>

- 불필요한 파티션이 있으면
```
sudo wipefs -a /dev/sda  # 모든 파티션 정보 삭제
sudo parted /dev/sda mklabel gpt  # GPT 파티션 테이블 생성

# 추가 설정 사항 -> 파티션 삭제가 불가능한 경우
sudo umount /dev/sda1
sudo parted /dev/sda
rm 1
```  
<br>

- sda의 기존의 모든 파일 시스템 및 파티션을 삭제
```
sudo wipefs --all /dev/sda
```  
<br>

### 2. 새로운 LVM 생성
- 물리 볼륨(PV) 생성
```
sudo pvcreate /dev/sda
sudo pvs # PV 결과 확인
sudo vgs # VG 내역 확인
```  
  

- 볼륨 그룹(VG) 생성
```
sudo vgcreate vg_data /dev/sda
```  
  
- 논리 볼륨(LV) 생성
```
sudo lvcreate -l 100%FREE -n lv_storage vg_data
sudo lvdisplay # 결과 확인
```  
<br>

### 3. 마운트 설정
- EXT4 파일 시스템 생성
  -  AI 모델 학습 서버에서는 대부분 EXT4 또는 XFS를 사용
  -  대용량 병렬 I/O가 많은 환경에서는 XFS가 유리하지만, 일반적인 경우 EXT4가 충분히 빠르고 안정적
```
sudo mkfs.ext4 /dev/vg_data/lv_storage
```  
  

- 마운트할 디렉토리 생성
```
sudo mkdir -p /mnt/data
```  
  

- 논리 볼륨 마운트
```
sudo mount /dev/vg_data/lv_storage /mnt/data
df -h | grep data # 마운트 확인
```  
  
- 자동 마운트 설정(재부팅 후에도 유지)
```
echo "/dev/vg_data/lv_storage /mnt/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```  

  <br>
  
### 4. 추후 사용자 추가되며 LVM 설정 변경시
- 사용자별 논리 볼륨(LV) 생성 (예: user1=1TB, user2=2TB, 나머지=1TB)
```
sudo lvcreate -L 1T -n lv_user1 vg_data
sudo lvcreate -L 2T -n lv_user2 vg_data
sudo lvcreate -l 100%FREE -n lv_shared vg_data  # 남은 공간 할당
```  
- 파일 시스템 생성 및 마운트
```
sudo mkfs.ext4 /dev/vg_data/lv_user1
sudo mkfs.ext4 /dev/vg_data/lv_user2
sudo mkfs.ext4 /dev/vg_data/lv_shared

sudo mkdir /mnt/user1 /mnt/user2 /mnt/shared
sudo mount /dev/vg_data/lv_user1 /mnt/user1
sudo mount /dev/vg_data/lv_user2 /mnt/user2
sudo mount /dev/vg_data/lv_shared /mnt/shared
```  
- 자동 마운트 설정 (재부팅 후에도 유지)
```
echo "/dev/vg_data/lv_user1 /mnt/user1 ext4 defaults 0 2" | sudo tee -a /etc/fstab
echo "/dev/vg_data/lv_user2 /mnt/user2 ext4 defaults 0 2" | sudo tee -a /etc/fstab
echo "/dev/vg_data/lv_shared /mnt/shared ext4 defaults 0 2" | sudo tee -a /etc/fstab
```  

---   

<br>

## 4) 설치 후 다른 사용자 추가
```
sudo adduser user2  # user2 사용자 추가
sudo usermod -aG sudo user2  # sudo 권한 부여 (필요하면)
```  

<br>
<br>
  
# 3. AI 모델링 준비
## 0) 기본 시스템 설정 및 최적화
- 시스템 업데이트 및 업그레이드
```
sudo apt update && sudo apt upgrade -y
```  
  
- 필수 패키지 설치
```
sudo apt install -y build-essential dkms unzip net-tools htop tmux vim git curl wget
```  
  
---  
<br>

## 1) NVIDIA 드라이버 및 CUDA 설치
- 드라이버 확인
```
nvidia-smi
```  
  
- 설치되지 않았음 설치
```
sudo ubuntu-drivers devices #-> recommended 나온거 설치
sudo apt install -v nvidia-drivers-550
sudo reboot
```  
  
- CUDA & cuDNN 설치 [링크](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
  - Linux -> x86_64 -> Ubuntu -> 22.04 -> deb(local)
- CUDA 경로 확인
```
whereis cuda # /usr/local/cuda
ls /usr/local/
```  
  
- 환경 변수 설정
```
echo 'export PATH=/usr/local/(cuda 맞는 버전)/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/(cuda 맞는 버전)/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo reboot
```  
  
- 재확인 : nvcc --version

---   
<br>

## 2) AI 개발 환경 세팅 (PyTorch, TensorFlow, Docker 등)
### 1. Python & AI 환경 구축
- Python & pip 최신화
```
sudo apt install -y python3 python3-pip python3-venv
```  
  
- 가상 환경 설정 (venv) -> 선택
```
python3 -m venv ~/ai_env
source ~/ai_env/bin/activate
```  
  

- 필수 패키지 설치
```
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn jupyter tqdm scikit-learn
```  
  
   <br>
   
### 2. Jupyter Notebook & VS Code 설정
- 아나콘다 설치 [설치 링크](https://www.anaconda.com/download/success)
  - 64-Bit (x86) Installer 선택
```
cd ~/Downloads
bash Anaconda3-2023.03-Linux-x86_64.sh # 설치
source ~/.bashrc # 환경변수 설정
```  
  
- Jupyter Notebook 설치
```
pip install jupyterlab
```  
  

- 백그라운드 실행
```
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```  
  

- VS Code 설치 [설치 링크](https://code.visualstudio.com/)
  - .deb 파일을 다운로드
  - 다운로드 경로로 이동
```
sudo apt install ./(설치경로)/(설치파일명)

sudo apt install ./Downloads/code_1.98.1-1741624510_amd64.deb
```  
  

<br>

### 3. PyTorch & TensorFlow 설치
- PyTorch (CUDA 지원)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```    
  
- TensorFlow (GPU 지원)
```
pip install tensorflow
```    
  
- 설치 확인
```
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```   
  
---  

<br>

## 3) 시스템 성능 최적화
- SWAP 메모리 늘리기
```
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```  
  
- CPU 성능 최적화
```
sudo apt install -y cpufrequtils
sudo cpufreq-set -g performance
```  
  
- 네트워크 속도 최적화
```
sudo sysctl -w net.ipv4.tcp_window_scaling=1
sudo sysctl -w net.ipv4.tcp_sack=1
```  
  
---  
<br>

## 4) 보안 및 원격 접속 설정
### 1. SSH 설정
- SSH 서버 설치
```
sudo apt update
sudo apt install openssh-server
sudo systemctl status ssh
```  
  
- SSH 포트 번호 변경 및 업데이트 -> **9972**
```
sudo vi /etc/ssh/sshd_config
sudo systemctl restart sshd
```  
  
- **맥북 터미널에 입력 : ssh -p 9972 hyebit@121.140.74.6**
  
- **만약 WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! 발생시**
```
cd /Users/hyebit/     
ls # ssh 확인

cd .ssh
ls # 파일 확인

# known으로 시작하는 파일을 지워야 함
rm known_hosts
rm known_hosts.old
rm config

clear
```  
  
- 포트포워딩 등록 및 유지(설정하지 않음)
```
sudo iptables -A INPUT -p tcp --dport 9972 -j ACCEPT

# 재부팅 시 유지
sudo apt install iptables-persistent
```   
  
- 방화벽 설정 (설정 안함)
```
sudo ufw allow 22/tcp
sudo ufw allow 8888/tcp
sudo ufw enable
```  
  
  <br>
  
### 2. iptime 앱 설정
- ssh 네트워크 인터페이스 이름 확인
```
ifconfig
```

<img width="817" alt="image" src="https://github.com/user-attachments/assets/d0281fc4-cbe8-4643-b730-68220adcaffb" />   
  
- WOL 패키지 설치
```
sudo apt-get install net-tools ethtool wakeonlan
```  
  
- 네트워크 인터페이스 설정 파일 변경
```
sudo vi /etc/network/interfaces

# 내용 추가
# post-up /sbin/ethtool -s (네트워크 인터페이스 이름) wol g
# post-down /sbin/ethtool -s (네트워크 인터페이스 이름) wol g
```  
  
- 서비스 등록 설정
```
sudo vi /etc/systemd/system/wol.service

# 내용 추가
# [Unit]
# Description=Configure Wake-up on LAN

# [Service]
# Type=oneshot
# ExecStart=/sbin/ethtool -s [인터페이스명] wol g

# [Install]
# WantedBy=basic.target
```  
  
- 서비스 시작
```
sudo systemctl enable /etc/systemd/system/wol.service
sudo systemctl start wol.service
```  
  

---   

<br>

## 5) Docker & ML Ops 환경 구축
-  Docker 설치
```
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```  
  
- NVIDIA-Docker 추가
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
    && sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```   
  
---  

<br>

## 6) Git 연결
### 1. Git 설치 확인 및 설치
- 설치 확인
```
git --version
```  
  
- 설치
```
sudo apt update && sudo apt install git -y
```  

<br>
    
### 2. GitHub 계정 및 SSH 키 생성
- SSH 키 생성
```
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```  
  
- SSH 에이전트 실행 후 키 추가
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```    
  
- SSH 키 확인 및 복사
```
cat ~/.ssh/id_rsa.pub
```   

<br>
    
### 3. GitHub에 SSH 키 등록  
1️⃣ [GitHub SSH 키 설정 페이지](https://github.com/settings/keys)로 이동  
2️⃣ “New SSH key” 클릭  
3️⃣ “Title”에 서버 이름 입력 (예: “Ubuntu Server”)  
4️⃣ “Key” 부분에 복사한 공개 키 붙여넣기  
5️⃣ “Add SSH key” 클릭하여 저장  

<br>

### 4. SSH 연결 테스트
```
ssh -T git@github.com
```  
  
- 성공 : Hi <GitHub 사용자명>! You've successfully authenticated, but GitHub does not provide shell access.

<br>

### 5. Git 설정 (최초 1회)
- 사용자 정보 저장
```
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```   
  
- 적용 확인
```
git config --list
```   

<br>

### 6. GitHub 저장소와 연결
- GitHub에서 Clone (기존 프로젝트 가져오기)
```
git clone git@github.com:사용자명/저장소이름.git
cd 저장소이름
```  
  
- GitHub 새 저장소 생성 후 Push (신규 프로젝트 업로드)
```
# Git 초기화
git init

# 파일 추가
git add .

# 커밋 생성
git commit -m "첫 커밋: Ubuntu 서버에서 Git 설정 완료"

# GitHub 원격 저장소 연결
git remote add origin git@github.com:사용자명/저장소이름.git

# 원격 저장소로 Push
git branch -M main
git push -u origin main

```   
 
<br>
<br>

# 기타
## Q. GRUB Menu 페이지는 왜 없어지지 않을까?
- GRUB는 일부러 보이게 하기도 한다

- [우분투 20.04에서 Grub 편집: grub-customizer 와 Grub theme 설정](https://kibua20.tistory.com/128)
    - 위의 링크처럼 테마를 변경하기도

## Ubuntu의 절전 모드 또는 화면 꺼짐 설정
- 절전 모드 및 화면 꺼짐 비활성화
```
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' -> 이것만 설정함
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.session idle-delay 0
```  
  
---   
<br>

## 한글 입력 설정
- Setting에서 Region & Language 탭으로 이동한 후 **[Manage Installed Languages]**를 클릭
- 팝업이 뜰 텐데 그냥 **[Install]**
- 알아서 필요한 파일들을 설치
```
sudo reboot
```  
  
- Setting에서 Keyboard 탭으로 이동해서 [+]을 클릭 후 [Korean]을 선택하면 재부팅 전에는 없던 "Korean (Hangul)"이 생긴 것을 볼 수 있다. 이것을 클릭 후 추가
- Korean (Hangul)의 [Preferences]를 선택
- Toggle Key들을 제거
- Toggle Key를 추가해주기 위해 Add를 누른 뒤에 "한영키"(여기서는 Alt_R로 인식)를 한번만 클릭



---   
<br>

## AI 모델링에서 SSD + HDD를 효율적으로 사용
- SSD → AI 모델 학습 중 임시 데이터, 캐시, 체크포인트 저장
- HDD → 데이터셋, 로그, 결과 저장

- 예시
  | **사용 목적**                | 💾 **SSD (`/`)** | 📂 **HDD (`/mnt/data/`)** |
  |-----------------------------|----------------|----------------|
  | **AI 코드 실행 & 모델 학습** | ✅ `/home/username/AIproject/` | ❌ 사용하지 않음 |
  | **데이터셋 저장 (Raw Data)** | ❌ 사용하지 않음 | ✅ `/mnt/data/dataset/` |
  | **학습 중간 결과 저장 (체크포인트)** | ✅ `/tmp/model_checkpoint.pth` | ❌ 사용하지 않음 |
  | **학습 로그 (TensorBoard, MLflow)** | ✅ `/tmp/tensorboard_logs/` | ✅ `/mnt/data/logs/` |
  | **최종 모델 저장** | ❌ 사용하지 않음 | ✅ `/mnt/data/models/final_model.pth` |
  | **결과 분석 및 시각화 데이터** | ✅ `/tmp/results/` | ✅ `/mnt/data/outputs/` |
  <br>

### PyTorch
```py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# HDD에 데이터셋 저장
data_dir = "/mnt/data/dataset"

# SSD에 캐시 저장 (학습 속도 최적화)
cache_dir = "/tmp/pytorch_cache"

# 데이터셋 로드
dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transforms.ToTensor()
)

# 데이터로더 생성
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 모델 체크포인트 저장 (SSD 사용)
torch.save(model.state_dict(), "/tmp/model_checkpoint.pth")

# 최종 모델 저장 (HDD 사용)
torch.save(model.state_dict(), "/mnt/data/models/final_model.pth")
```

  <br>
  
### TensorFlow
```py
import tensorflow as tf

# 데이터셋 경로 설정 (HDD 사용)
dataset_path = "/mnt/data/dataset"

# 모델 체크포인트 경로 (SSD 사용)
checkpoint_path = "/tmp/tf_checkpoint"

# 최종 모델 저장 경로 (HDD 사용)
final_model_path = "/mnt/data/models/final_model"

# 데이터 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(224, 224),
    batch_size=32
)

# 모델 저장 콜백 (SSD 체크포인트)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)

# 모델 학습
model.fit(train_ds, epochs=10, callbacks=[checkpoint_callback])

# 최종 모델 저장 (HDD)
model.save(final_model_path)
```
