## 데이터 전처리

### moviepy
'''
- moviepy 라이브러리를 사용하여 비디오 파일에서 오디오를 추출하고, 해당 오디오를 MP3 파일로 저장
- 대량의 파일을 처리하거나 복잡한 작업에서는 FFmpeg보다 성능이 떨어질 수 있음
- 사용이 간편함
- 2.3초 소요
'''

from moviepy.editor import VideoFileClip

# 처리하고자 하는 비디오 파일 경로
video_file_pth = ('./video/관객이 될게 최최종.mp4')

# 해당 경로에서 비디오 파일 로드
video = VideoFileClip(video_file_pth)

# 저장하고자 하는 오디오 파일 경로
audio_file_pth = ('./audio/관객이 될게 최최종_moviepy.mp3')

# video.audio - 비디오 객체에서 오디오 클립 추출
# write_audiofile() - 오디오 클립을 지정한 경로에 저장
video.audio.write_audiofile(audio_file_pth)

# 저장한 MP3 파일을 바이너리 읽기 모드로 열기
# 바이너리 모드에서는 파일 내용을 원시 바이트 데이터로 처리
# 파일의 내용을 그대로 바이트 형태로 읽거나 쓸 수 있으며, 텍스트 인코딩과는 무관하게 파일 다룸
# 이미지 파일, 오디오 파일, 비디오 파일과 같은 비텍스트 파일을 처리할 때 유용
audio_file = open('./audio/관객이 될게 최최종.mp3', 'rb')



### FFmpeg
'''
- 매우 강력한 멀티미디어 프레임워크로, 커맨드라인 기반으로 거의 모든 멀티미디어 작업을 지원
- 오픈소스 멀티미디어 프레임워크로, 비디오, 오디오, 이미비 파일을 변환, 인코딩, 디코딩, 스트리밍하는 데 사용
- 자동화된 작업에서 많이 사용
- 단점 : 명령어가 복잡할 수 있음
- 2.1초 소요
'''

import subprocess

# 비디오와 오디오 파일 경로 설정
video_file_pth = './video/관객이 될게 최최종.mp4'
audio_file_pth = './audio/관객이 될게 최최종_ffmpeg.mp3'

# FFmpeg 명령어 실행
subprocess.run([
    'ffmpeg',
    '-i', video_file_pth,  # 입력 비디오 파일
    '-q:a', '0',           # 최고 품질의 오디오
    '-map', 'a',           # 오디오 스트림만 추출
    audio_file_pth         # 출력 오디오 파일
], check=True)

# 오디오 파일을 바이너리 모드로 열어 크기 출력
with open(audio_file_pth, 'rb') as audio_file:
    print(f"Audio file size: {len(audio_file.read())} bytes")


## 오디오 파일 텍스트로 변환
import whisper

model = whisper.load_model('base')
result_moviepy = model.transcribe('./audio/관객이 될게 최최종_moviepy.mp3')
result_ffmpeg = model.transcribe('./audio/관객이 될게 최최종_ffmpeg.mp3')

result_moviepy['text']
result_ffmpeg['text']
