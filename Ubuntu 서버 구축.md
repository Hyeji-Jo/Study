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



# 2. Ubuntu 22.04 버전 설치

## 1) Ubuntu 초기화  부팅 USB 활용

- Del 버튼으로 Bios 진입  
    - **BIOS 설정 확인**  
        - CSM(Compatibility Support Module): Disabled 유지 (UEFI 모드에서 설치)  
        - Secure Boot: Disabled로 변경 (NVIDIA 드라이버 및 CUDA 설치를 위해 필요)  
        - Boot Option #1: UEFI: SanDisk, Partition 1 (SanDisk)로 변경  

<img width="539" alt="image" src="https://github.com/user-attachments/assets/a1b8b3ec-3960-4692-a1b1-751f29c135b6" />
- USB 부팅을 통해 우분투를 새로 설치하려는 경우 = **Yes** -> **“Try or Install Ubuntu”를 선택한 후 엔터**


## 2) 설치된 후 처음 부팅된 화면에서 옵션 선택  

### 1. 언어 및 키보드 설정 → English로 하는게 AI 모델링 시 문제가 별로 발생되지 않음

- 언어 선택: **English** (또는 원하는 언어)  
- 키보드 레이아웃: 기본값 **English (US)** 유지 (다른 키보드 레이아웃이 필요하면 변경)  
![image](https://github.com/user-attachments/assets/c2b94210-5105-406f-8879-8fdf17152c67)
- USB 부팅을 통해 우분투를 새로 설치하려는 경우 = **Yes**  → **“Try or Install Ubuntu”를 선택한 후 엔터  **

---

### 2. 네트워크 설정

- **DHCP 자동 설정:** 기본값을 유지하면 자동으로 IP를 가져옵니다.  
- **수동 설정이 필요한 경우:** Manual을 선택하고 IP, 서브넷 마스크, 게이트웨이 등을 입력  
- 네트워크가 필요하지 않으면 *Do not configure network”를 선택하고 넘어가도 됩니다.  

---

### 3. 설치 유형 선택 

- **“Use entire disk and set up LVM”** → 이 옵션을 선택하면 **기존 파티션이 모두 삭제**되고 LVM으로 설정됩니다.  
- 만약 **LVM을 수동 설정하고 싶다면**, “Custom storage layout”을 선택합니다.
<img width="1245" alt="image" src="https://github.com/user-attachments/assets/2d900132-e8e5-4e70-b37f-0cff14409a2a" />


<img width="1194" alt="image" src="https://github.com/user-attachments/assets/16fd5da5-208c-4ea6-a60e-80e7ce5c4c1e" />
- 위에는 **SSD에** 설치    
    - NVMe SSD는 HDD보다 훨씬 빠른 속도(읽기/쓰기 성능)가 나오므로 운영체제(OS), 소프트웨어, AI 모델 실행에 최적
	- OS를 HDD에 설치하면 부팅과 작업 속도가 느려짐
- 아래는 HDD에 설치


---

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

---


### 5. 사용자 계정 및 SSH 설정

- 관리자 계정 생성 (admin 또는 사용자 지정 계정)  -> **원하는 이름 작성**
- **SSH 설치**: **“Install OpenSSH server”** 옵션을 체크 (리모트 접속 필요 시 필수)  

---

### 6. 우분투 설치 및 부팅 설정

- 모든 설정을 확인하고 **설치를 진행**  
- 설치 완료 후, 재부팅하면 새롭게 설정된 LVM 기반 우분투 서버로 진입 가능  



## 3) HDD LVM(Logical Volume Manager) 활용 (유연한 용량 조정) -> 1,2번중 선택해서 진행
### 1. HDD(4TB) 마운트 확인 및 설정
- 마운트 확인
```
lsblk
```  
-> /dev/sda 또는 /dev/sdb로 표시될 가능성이 높음.  

- HDD를 /mnt/data 디렉토리에 마운트
```
sudo mkdir -p /mnt/data
sudo mount /dev/sda1 /mnt/data
```

- 부팅 시 자동마운트 설정
```
echo "/dev/sda1 /mnt/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

### 2. HDD LVM 설정
- HDD를 LVM 볼륨 그룹(VG)으로 설정
```
sudo pvcreate /dev/sda
sudo vgcreate vg_data /dev/sda
```
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

## 4) 설치 후 다른 사용자 추가
```
sudo adduser user2  # user2 사용자 추가
sudo usermod -aG sudo user2  # sudo 권한 부여 (필요하면)
```



# 3. AI 모델링 준비
## 1) NVIDIA 드라이버 및 CUDA 설치

## 2) AI 개발 환경 세팅 (PyTorch, TensorFlow, Docker 등)

## 3) SSH 원격 접속 설정 (서버 운영 시 필수)


## 4)
