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


![image.png](attachment:d9a54cf4-3bd2-4633-bed0-21079cb0547c:image.png)

![image.png](attachment:086b721d-6665-474b-9813-e2cdedaad2c8:image.png)

# 2. Ubuntu 22.04 버전 설치

## 1) Ubuntu 초기화  부팅 USB 활용

- Del 버튼으로 Bios 진입  
    - **BIOS 설정 확인**  
        - CSM(Compatibility Support Module): Disabled 유지 (UEFI 모드에서 설치)  
        - Secure Boot: Disabled로 변경 (NVIDIA 드라이버 및 CUDA 설치를 위해 필요)  
        - Boot Option #1: UEFI: SanDisk, Partition 1 (SanDisk)로 변경  

![image.png](attachment:64fcf8e0-ec01-46ac-98d1-cae6003c663f:image.png)

## 2) 설치된 후 처음 부팅된 화면에서 옵션 선택  

**1. 언어 및 키보드 설정 → English로 하는게 AI 모델링 시 문제가 별로 발생되지 않음**  

- 언어 선택: **English** (또는 원하는 언어)  
- 키보드 레이아웃: 기본값 **English (US)** 유지 (다른 키보드 레이아웃이 필요하면 변경)  

---

**2. 네트워크 설정**  

- **DHCP 자동 설정:** 기본값을 유지하면 자동으로 IP를 가져옵니다.  
- **수동 설정이 필요한 경우:** Manual을 선택하고 IP, 서브넷 마스크, 게이트웨이 등을 입력  
- 네트워크가 필요하지 않으면 *Do not configure network”를 선택하고 넘어가도 됩니다.  

---

**3. 설치 유형 선택**  

- **“Use entire disk and set up LVM”** → 이 옵션을 선택하면 **기존 파티션이 모두 삭제**되고 LVM으로 설정됩니다.  
- 만약 **LVM을 수동 설정하고 싶다면**, “Custom storage layout”을 선택합니다.  

---

**4. 디스크 파티션 설정 (LVM 구성)**  

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

- USB 부팅을 통해 우분투를 새로 설치하려는 경우 = Yes  

→ “Try or Install Ubuntu”를 선택한 후 엔터  

**5. 사용자 계정 및 SSH 설정**  

- 관리자 계정 생성 (admin 또는 사용자 지정 계정)  
- **SSH 설치**: **“Install OpenSSH server”** 옵션을 체크 (리모트 접속 필요 시 필수)  

---

**6. 우분투 설치 및 부팅 설정**  

- 모든 설정을 확인하고 **설치를 진행**  
- 설치 완료 후, 재부팅하면 새롭게 설정된 LVM 기반 우분투 서버로 진입 가능  
