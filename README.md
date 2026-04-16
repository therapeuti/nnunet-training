# nnU-Net v2 학습 가이드

이 저장소는 `nnU-Net v2`로 신장(`kidney`) 및 종양(`tumor`) 분할 모델을 학습하기 위한 데이터셋과 실행 환경을 정리한 프로젝트입니다.

현재 데이터셋:

```text
Dataset001_mydata
```


## 1. nnU-Net이 무엇인가

공식 `MIC-DKFZ/nnUNet` 저장소에 따르면, `nnU-Net`은 데이터셋을 자동으로 분석해서 전처리, 네트워크 설정, 학습, 모델 선택, 추론 파이프라인을 자동으로 맞춰주는 의료영상 분할 프레임워크입니다.

핵심 개념:
- 의료영상 semantic segmentation에 특화된 프레임워크
- 데이터셋 특성에 맞게 preprocessing과 network configuration을 자동 설정
- 사용자는 보통 모델 코드를 직접 작성하기보다, 데이터셋 형식을 맞춘 뒤 명령어로 실행

대표적인 구성:
- `2d`
- `3d_fullres`
- `3d_lowres -> 3d_cascade_fullres`

다만 모든 데이터셋에서 모든 구성이 생성되는 것은 아니며, 데이터 특성에 따라 필요한 구성만 자동으로 만들어집니다.


## 2. 개발환경 세팅

공식 문서에서는 먼저 하드웨어에 맞는 `PyTorch`를 설치한 뒤, 그 다음 `nnunetv2`를 설치하라고 안내합니다.

권장 환경:
- Windows 10 또는 Windows 11
- Python 3.9 이상
- CUDA 지원 GPU 권장
- SSD 사용 권장

공식 문서에서는 학습용 GPU 메모리를 최소 10GB 이상 권장합니다.


### 2-1. 가상환경 생성

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```


### 2-2. CUDA 버전에 맞는 PyTorch 설치

먼저 GPU 상태를 확인합니다.

```cmd
nvidia-smi
```

그 다음 PyTorch 공식 설치 페이지에서 본인 환경에 맞는 명령을 선택합니다.

공식 페이지:
- https://pytorch.org/get-started/locally/

예시:

CUDA 12.1

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU 전용

```cmd
pip install torch torchvision torchaudio
```

참고:
- nnU-Net 공식 README에는 `torch 2.9.0`에서 3D convolution과 AMP 관련 성능 저하가 있다고 적혀 있습니다.
- 해당 문제가 있다면 `torch 2.8.x 이하`를 쓰는 것이 안전합니다.


### 2-3. PyTorch CUDA 인식 확인

```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

예시:

```text
True
12.1
```

의미:
- 첫 줄 `True`: CUDA 사용 가능
- 둘째 줄: PyTorch가 인식한 CUDA 버전


### 2-4. nnU-Net v2 설치

공식 저장소 기준 기본 설치:

```cmd
pip install nnunetv2
```

설치 확인:

```cmd
nnUNetv2_plan_and_preprocess --help
nnUNetv2_train --help
```


## 3. nnU-Net 데이터셋 구조

공식 문서에 따르면 데이터셋은 `nnUNet_raw` 아래에 dataset ID와 dataset name을 포함한 폴더로 저장합니다.

예:

```text
nnUNet_raw/
├─ Dataset001_mydata
├─ Dataset002_xxx
├─ Dataset003_xxx
```

현재 프로젝트에서 사용하는 데이터셋 경로:

```text
nnUnet_raw/Dataset001_mydata/
```


### 3-1. 필수 폴더 구조

`Dataset001_mydata` 내부는 아래처럼 구성되어 있어야 합니다.

```text
nnUnet_raw/
└─ Dataset001_mydata/
   ├─ dataset.json
   ├─ imagesTr/
   ├─ imagesTs/
   └─ labelsTr/
```

각 폴더 의미:
- `imagesTr`: 학습 이미지
- `labelsTr`: 학습 라벨
- `imagesTs`: 테스트 이미지 저장용, 선택 사항
- `dataset.json`: 데이터셋 메타데이터

공식 문서에 따르면 `imagesTs`는 nnU-Net 학습에 직접 사용되지 않으며, 사용자가 테스트 이미지를 보관하는 용도로 둘 수 있습니다.


### 3-2. 파일명 규칙

공식 문서에 따르면 입력 채널은 파일명 끝의 4자리 채널 번호로 구분합니다.

이미지 파일 규칙:

```text
{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}
```

라벨 파일 규칙:

```text
{CASE_IDENTIFIER}.{FILE_ENDING}
```

현재 프로젝트는 CT 단일 채널만 사용하므로 `_0000`만 사용합니다.

예시:

```text
imagesTr/case_00001_0000.nii.gz
imagesTr/case_00002_0000.nii.gz

labelsTr/case_00001.nii.gz
labelsTr/case_00002.nii.gz
```

즉 같은 케이스는 아래처럼 매칭됩니다.

```text
imagesTr/case_00001_0000.nii.gz
labelsTr/case_00001.nii.gz
```


### 3-3. 현재 프로젝트의 라벨 정의

현재 라벨 정의는 다음과 같습니다.

- `0`: background
- `1`: kidney
- `2`: tumor

공식 문서에서는 다음을 요구합니다.
- background는 반드시 `0`
- segmentation은 정수형 semantic class map
- class 값은 연속적이어야 함 (`0,1,2,3,...`)


## 4. dataset.json 구성 방법

현재 파일 경로:

```text
nnUnet_raw/Dataset001_mydata/dataset.json
```

현재 프로젝트에서 사용하는 설정:

```json
{
  "channel_names": {
    "0": "CT"
  },
  "labels": {
    "background": 0,
    "kidney": 1,
    "tumor": 2
  },
  "numTraining": 318,
  "file_ending": ".nii.gz",
  "name": "mydata",
  "overwrite_image_reader_writer": "NibabelIOWithReorient"
}
```


### 4-1. 항목 설명

`channel_names`
- 채널 번호와 의미를 연결
- 현재는 단일 CT 채널이므로 `"0": "CT"`

`labels`
- 클래스 이름과 정수 라벨 값 정의
- 현재는 `background`, `kidney`, `tumor`

`numTraining`
- 실제 `imagesTr`, `labelsTr`에 들어간 학습 케이스 수
- 현재는 `318`

`file_ending`
- 현재 데이터 형식은 `.nii.gz`

`name`
- 데이터셋 이름
- 현재는 `"mydata"`

`overwrite_image_reader_writer`
- 공식 문서에서 여러 reader/writer를 지원하며, 현재 프로젝트는 `NibabelIOWithReorient`를 사용


### 4-2. 작성 시 주의점

- `numTraining`은 실제 파일 개수와 일치해야 합니다.
- 이미지와 라벨의 geometry는 같은 케이스 내에서 서로 일치해야 합니다.
- 모든 케이스는 같은 채널 구성을 가져야 합니다.
- inference 때도 training 때와 같은 채널 순서를 유지해야 합니다.


## 5. 현재 프로젝트의 데이터셋 상태

현재 프로젝트는 `A phase` CT만 학습에 사용하도록 구성되어 있습니다.

현재 준비 상태:
- `segmentation_A`가 있는 케이스: `318개`
- `image_A`가 있는 케이스: `318개`
- `imagesTr`: `318개`
- `labelsTr`: `318개`
- `dataset.json`의 `numTraining`: `318`

즉 현재 상태에서는 `Dataset001_mydata`가 바로 `nnU-Net v2` 학습에 들어갈 수 있는 상태입니다.


## 6. 환경변수 설정

공식 문서에 따르면 `nnU-Net`은 아래 3개 환경변수를 사용합니다.

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

이 변수들은 raw data, preprocessed data, trained model 위치를 nnU-Net에 알려줍니다.


### 6-1. CMD에서 임시 설정

공식 문서의 Windows 예시를 현재 프로젝트 경로에 맞게 바꾸면 다음과 같습니다.

```cmd
set nnUNet_raw=C:\Users\user\nnunet\nnUnet_raw
set nnUNet_preprocessed=C:\Users\user\nnunet\nnUnet_preprocessed
set nnUNet_results=C:\Users\user\nnunet\nnUnet_results
```

확인:

```cmd
echo %nnUNet_raw%
echo %nnUNet_preprocessed%
echo %nnUNet_results%
```

주의:
- 이 방식은 현재 `cmd` 세션에만 적용됩니다.
- 창을 닫으면 다시 설정해야 합니다.


### 6-2. CMD에서 영구 설정

영구 설정이 필요하면 `setx`를 사용할 수 있습니다.

```cmd
setx nnUNet_raw "C:\Users\user\nnunet\nnUnet_raw"
setx nnUNet_preprocessed "C:\Users\user\nnunet\nnUnet_preprocessed"
setx nnUNet_results "C:\Users\user\nnunet\nnUnet_results"
```

주의:
- `setx` 실행 후에는 새 `cmd` 창을 다시 열어야 반영됩니다.


## 7. 학습 실행 순서

공식 nnU-Net 흐름은 보통 다음 순서입니다.

1. 데이터셋 준비
2. preprocessing + experiment planning
3. training

현재 프로젝트는 1단계가 끝난 상태이므로, 실제로는 2단계부터 시작하면 됩니다.


### 7-1. 전처리 및 무결성 검사

```cmd
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

설명:
- `-d 1`: dataset ID 1, 즉 `Dataset001_mydata`
- `--verify_dataset_integrity`: 파일 구조, geometry, 개수 등을 확인


### 7-2. 학습 시작

가장 기본적인 예시:

```cmd
nnUNetv2_train 1 3d_fullres 0
```

설명:
- `1`: dataset ID
- `3d_fullres`: 3D full resolution configuration
- `0`: fold 0


### 7-3. 전체 fold 학습

보통 cross-validation을 위해 fold 0~4를 각각 학습합니다.

```cmd
nnUNetv2_train 1 3d_fullres 0
nnUNetv2_train 1 3d_fullres 1
nnUNetv2_train 1 3d_fullres 2
nnUNetv2_train 1 3d_fullres 3
nnUNetv2_train 1 3d_fullres 4
```


## 8. 현재 프로젝트에서 바로 실행할 명령

`cmd` 기준으로 가장 바로 사용할 수 있는 순서는 아래와 같습니다.

```cmd
set nnUNet_raw=C:\Users\user\nnunet\nnUnet_raw
set nnUNet_preprocessed=C:\Users\user\nnunet\nnUnet_preprocessed
set nnUNet_results=C:\Users\user\nnunet\nnUnet_results

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train 1 3d_fullres 0
```

fold 0 학습이 잘 되면 이후 fold 1~4를 이어서 실행하면 됩니다.


## 9. 참고 자료

공식 nnU-Net 저장소 및 문서:
- https://github.com/MIC-DKFZ/nnUNet
- https://github.com/MIC-DKFZ/nnUNet/blob/master/readme.md
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md
