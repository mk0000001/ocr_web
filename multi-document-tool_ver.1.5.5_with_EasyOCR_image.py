#python -m venv {name}
version  = 'multi document tool_ver.1.5.5 with EasyOCR'

import os
import shutil
# import twain
import tkinter as tk
# import win32printing as pr
from tkinter import ttk
import pytesseract
import cv2 #opencv-python
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
# from PIL import Image
import zipfile
from datetime import datetime
import time
from multiprocessing import Process, Queue
import threading
import easyocr
import fitz #-> PyMuPDF
import torch # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# import datetime

auto = {'auto' : 'true'}
engine = {'engine' : 'EasyOCR'}

def rename_files(start_number, dest_folder): # 홀수 파일 이름 변경 메서드6
    counter = start_number
    progress_var.set(0)

    file_list = os.listdir(os.path.join(program_folder, source_folder)) #소스 폴더의 파일들의 이름을 확장자까지 리스트로 묶는 코드
    total_files = len(file_list) # 소스파일의 총 파일 개수

    for index, filename in enumerate(file_list): # 파일 리스트의 각 항목의 인덱스와 파일 이름을 index, filename에 저장
        file_extension = os.path.splitext(filename)[1] # 파일 확장자가 포함된 파일 이름에서 확장자를 분리
        new_name = f"{counter}{file_extension}" # 지정할 파일 이름
        source_file = os.path.join(program_folder, source_folder, filename) # 소스 폴더 정의
        destination_file = os.path.join(program_folder, dest_folder, new_name) # 새로운 속성의 파일 객체 정의

        shutil.copy(source_file, destination_file)  # 파일 복사, 소스 폴더의 파일을 새로운 속성의 파일 객체로 복사

        counter += 2  # 다음 홀수/짝수 번호로 업데이트

        progress_var.set((index + 1) / total_files * 100) # 작업 진행률 변경
        root.update_idletasks()

def rename_files_reverse(start_number, dest_folder): #rename_files와 file_list.sort(reverse=True) 를 제외한 모든 코드 동일
    counter = start_number
    progress_var.set(0)

    file_list = os.listdir(os.path.join(program_folder, source_folder))
    file_list.sort(reverse=True) # rename_files의 메서드에서 파일 이름을 담은 리스트를 역순으로 정렬
    total_files = len(file_list)

    for index, filename in enumerate(file_list):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{counter}{file_extension}"
        source_file = os.path.join(program_folder, source_folder, filename)
        destination_file = os.path.join(program_folder, dest_folder, new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사

        counter += 2  # 다음 홀수/짝수 번호로 업데이트

        progress_var.set((index + 1) / total_files * 100)
        root.update_idletasks()

def search(text):
    file_list = os.listdir(os.path.join(program_folder, "OCR"))
    print(file_list)
    list = []
    result = []
    a = 0
    for index, filename in enumerate(file_list):
        f = open(os.path.join(program_folder, "OCR") + os.sep + filename, 'r', encoding='utf-8')
        total_files = len(file_list)
        data = f.read()
        print(data)
        lines = data.splitlines()
        # f.close()
        print("lines : {}".format(lines))
        for line in lines:
            if line.find(text) != -1:
                print("find : {}".format(line.find(text)))
                if a == 0:
                    list.append(filename)
                    a += 1
                for i in range(1, len(list) + 1):
                    if filename != list[i - 1]:
                        list.append(filename)
            print(line, text, text in line)
        f.close()
    for i in list:
        if i not in result:
            result.append(i)
    result.sort()
    return(result)

def num_sorting(list):
    final_list = []
    for i in list:
        x = []
        y = ''
        for j in i:
            if j == '0':
                x.append('0')
            elif j == '1':
                x.append('1')
            elif j == '2':
                x.append('2')
            elif j == '3':
                x.append('3')
            elif j == '4':
                x.append('4')
            elif j == '5':
                x.append('5')
            elif j == '6':
                x.append('6')
            elif j == '7':
                x.append('7')
            elif j == '8':
                x.append('8')
            elif j == '9':
                x.append('9')
        for k in range(len(x)):
            y += x[k]
        final_list.append(y)
    # final_list.sort()
    list_dictionary = {}
    for l in range(len(list)):
        list_dictionary[final_list[l]] = list[l]

    sorted_final_list = final_list
    sorted_final_list.sort()
    
    return(sorted_final_list, final_list, list_dictionary)

def automatic_Sorting(start_number, start_number2, dest_folder, dest_folder2, txt):
    counter = start_number
    counter2 = start_number2
    progress_var.set(0)

    file_list = os.listdir(os.path.join(program_folder, source_folder))
    print(file_list)
    backup_file_list = file_list
    # file_list.sort()

    sorted_final_list, final_list, list_dictionary = num_sorting(file_list)

    sorted_file_list = []

    for i in range(len(sorted_final_list)):
        sorted_file_list.append(list_dictionary.get(sorted_final_list[i]))

    file_list = sorted_file_list
    print(file_list)

    # sort_file = file_list
    # sort_file.sort()
    # print(sort_file)
    # file_list.sort()
    total_files = len(file_list)
    # print(file_list)

    max_length = max(len(f) for f in file_list)

    # 모든 파일 이름을 동일한 길이로 만듦
    file_list_names = [f.zfill(max_length) for f in file_list]

    # print(file_list)
    # print(file_list_names)
    #
    # print(file_list/.lnjhuy8u6pky[=no]m;
    # message["text"] = ""
    time.sleep(1)
    aver_time = []

    for index, filename in enumerate(file_list):
        # file_extension = os.path.splitext(filename)[1]
        start_time = time.time()
        new_name = file_list_names[index]

        source_file = os.path.join(program_folder, source_folder, filename)
        destination_file = os.path.join(program_folder, "process", new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사
        end_time = time.time()

        progress_var.set((index + 1) / total_files * 100)

        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files
        message["text"] = f"파일 복사 작업 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        root.update_idletasks()

    file_list = os.listdir(os.path.join(program_folder, "process"))

    # print(file_list)

    half_file_list = file_list[:(len(file_list) // 2)]
    half_file_list_end = file_list[len(half_file_list):]

    # print((half_file_list_end, half_file_list))
    aver_time.clear()

    for index, filename in enumerate(half_file_list):
        start_time = time.time()
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{txt}{counter}{file_extension}"
        source_file = os.path.join(program_folder, "process", filename)
        destination_file = os.path.join(program_folder, dest_folder, new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사

        counter += 2  # 다음 홀수/짝수 번호로 업데이트

        end_time = time.time()

        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files * 2
        message["text"] = f"작업 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set((index + 1) / total_files * 100)
        root.update_idletasks()

    half_file_list_end.sort(reverse=True)  # rename_files의 메서드에서 파일 이름을 담은 리스트를 역순으로 정렬

    aver_time.clear()
    for index, filename in enumerate(half_file_list_end):
        start_time = time.time()
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{txt}{counter2}{file_extension}"
        source_file = os.path.join(program_folder, "process", filename)
        destination_file = os.path.join(program_folder, dest_folder2, new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사

        counter2 += 2  # 다음 홀수/짝수 번호로 업데이트

        end_time = time.time()

        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files
        message["text"] = f"작업 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set(50 + ((index + 1) / total_files * 100))
        root.update_idletasks()

    aver_time.clear()
    for index, filename in enumerate(file_list):
        start_time = time.time()
        del_file = os.path.join(program_folder, "process", filename)
        os.remove(del_file)
        end_time = time.time()
        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files
        message["text"] = f"작업 파일 삭제 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set(((index + 1) / total_files * 100))
        root.update_idletasks()
    os.removedirs("process")

    message["text"] = "작업 완료"
    time.sleep(2)
    message["text"] = "대기 중"

def not_sort_automatic_Sorting(start_number, start_number2, dest_folder, dest_folder2, txt):
    counter = start_number
    counter2 = start_number2
    progress_var.set(0)

    file_list = os.listdir(os.path.join(program_folder, source_folder))
    print(file_list)
    # file_list.sort()
    # sort_file = file_list
    # sort_file.sort()
    # print(sort_file)
    # file_list.sort()
    total_files = len(file_list)
    # print(file_list)

    max_length = max(len(f) for f in file_list)

    # 모든 파일 이름을 동일한 길이로 만듦
    file_list_names = [f.zfill(max_length) for f in file_list]

    # print(file_list)
    # print(file_list_names)
    #
    # print(file_list/.lnjhuy8u6pky[=no]m;
    # message["text"] = ""
    time.sleep(1)
    aver_time = []

    for index, filename in enumerate(file_list):
        # file_extension = os.path.splitext(filename)[1]
        start_time = time.time()
        new_name = file_list_names[index]

        source_file = os.path.join(program_folder, source_folder, filename)
        destination_file = os.path.join(program_folder, "process", new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사
        end_time = time.time()

        progress_var.set((index + 1) / total_files * 100)

        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files
        message["text"] = f"파일 복사 작업 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        root.update_idletasks()

    file_list = os.listdir(os.path.join(program_folder, "process"))

    # print(file_list)

    half_file_list = file_list[:(len(file_list) // 2)]
    half_file_list_end = file_list[len(half_file_list):]

    # print((half_file_list_end, half_file_list))
    aver_time.clear()

    for index, filename in enumerate(half_file_list):
        start_time = time.time()
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{txt}{counter}{file_extension}"
        source_file = os.path.join(program_folder, "process", filename)
        destination_file = os.path.join(program_folder, dest_folder, new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사

        counter += 2  # 다음 홀수/짝수 번호로 업데이트

        end_time = time.time()

        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files * 2
        message["text"] = f"작업 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set((index + 1) / total_files * 100)
        root.update_idletasks()

    half_file_list_end.sort(reverse=True)  # rename_files의 메서드에서 파일 이름을 담은 리스트를 역순으로 정렬

    aver_time.clear()
    for index, filename in enumerate(half_file_list_end):
        start_time = time.time()
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{txt}{counter2}{file_extension}"
        source_file = os.path.join(program_folder, "process", filename)
        destination_file = os.path.join(program_folder, dest_folder2, new_name)

        shutil.copy(source_file, destination_file)  # 파일 복사

        counter2 += 2  # 다음 홀수/짝수 번호로 업데이트

        end_time = time.time()

        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files
        message["text"] = f"작업 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set(50 + ((index + 1) / total_files * 100))
        root.update_idletasks()

    aver_time.clear()
    for index, filename in enumerate(file_list):
        start_time = time.time()
        del_file = os.path.join(program_folder, "process", filename)
        os.remove(del_file)
        end_time = time.time()
        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files
        message["text"] = f"작업 파일 삭제 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set(((index + 1) / total_files * 100))
        root.update_idletasks()
    os.removedirs("process")

    message["text"] = "작업 완료"
    time.sleep(2)
    message["text"] = "대기 중"


def ocr(src_folder, engine):
    file_list = os.listdir(os.path.join(program_folder, src_folder))
    
    src_path = os.path.join(program_folder, src_folder)
    try:
        filename = file_list[0]
    except:
        error_popup("원본 파일 오류", f"❗ OCR 원본 폴더에 파일이 존재하지 않습니다. \n   {ocr_source_folder} 경로에 OCR 대상 파일을 넣어주십시오.")
    file_extension = os.path.splitext(filename)[1]

    tesseract_install = check_tesseract()
    
    if tesseract_install == False:
        error_popup("Tesseract Not Installed", "❗OCR 모듈인 Tesseract가 설치되지 않았습니다. \nTessract 모듈을 사용할 수 없습니다.")

    # print(file_extension)
    if file_extension == ".pdf" and file_extension == ".PDF":
        
        error_popup("파일 오류", "❗ 원본 파일의 확장자가 PDF입니다. 이미지 형식으로 바꿔주십시오.")
    else:
         if not os.path.exists(os.path.join("C:" + os.sep, "process")):
            message["text"] = "작업 폴더 생성 중"
            os.makedirs(os.path.join("C:" + os.sep, "process"))
         if not os.path.exists(os.path.join(program_folder, "OCR")):
            message["text"] = "작업 폴더 생성 중"
            os.makedirs(os.path.join(program_folder, "OCR"))
         ocr_file_list = os.listdir(os.path.join(program_folder, "OCR"))
        
         
         file_backup(os.path.join(program_folder, "OCR"),  "OCR - ")
         print(os.path.join(program_folder, "OCR"))
         for index, filename in enumerate(file_list):
            message["text"] = "파일 복사 중"
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{index}{file_extension}"
            total_files = len(file_list)
            source_file = os.path.join(program_folder, src_folder, filename)
            destination_file = os.path.join("C:" + os.sep, "process", new_name)

            shutil.copy(source_file, destination_file)  # 파일 복사

            progress_var.set((index + 1) / total_files * 100)

            root.update_idletasks()

         file_list = os.listdir(os.path.join(program_folder, "C:" + os.sep + "process"))
         message["text"] = f"문자 인식 작업 중 0%\n OCR 엔진 : {engine} \n 예상 시간 : 계산 중"

         print(engine)

         thread = threading.Thread(target=multi_OCR, name="thread", args=(file_list, total_files, engine))
         thread.start()
   
def multi_OCR(file_list, total_files, engine):
    torch.cuda.is_available() # cuda
    aver_time = []
    for index, file_name in enumerate(file_list):
        start_time = time.time()
        filePath = os.path.join("C:" + os.sep, "process", file_list[index])
        print(filePath)
        image = cv2.imread(filePath)

        # 이미지 전처리
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = binary

        print(torch.cuda.is_available()) #cuda
        if engine == 'Tesseract':
            ocr_engine = "Tesseract"
            text = pytesseract.image_to_string(image, lang='kor+eng')
        elif engine == 'EasyOCR':
            ocr_engine = "EasyOCR"
            reader = easyocr.Reader(['ko', 'en'], gpu=True)
            text = reader.readtext(image, detail=0)
        print(text)
    
        percent = round((index + 1) / total_files * 100, 1)
        progress_var.set((index + 1) / total_files * 100)
        f = open(program_folder + os.sep + "OCR" + os.sep +f"{index}.txt", 'w', encoding='utf-8')
        f.write(str(text))
        f.close
        end_time = time.time()
        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
    
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files * 2
        message["text"] = f"작업 중 {percent}%\n OCR 엔진 : {ocr_engine}\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        root.update_idletasks()
    message["text"] = "작업 파일 삭제 중"
    file_list = os.listdir(os.path.join("C:" + os.sep, "process"))
    for index, filename in enumerate(file_list):
        del_file = os.path.join("C:" + os.sep, "process", file_list[index])
        os.remove(del_file)
        progress_var.set((index + 1) / total_files * 100)
        root.update_idletasks()

    os.removedirs("C:" + os.sep + "process")
    message["text"] = "대기 중"
    
def check_tesseract():
    # pytesseract.pytesseract.tesseract_cmd의 기본값은 'tesseract'
    tesseract_cmd = 'tesseract'

    # shutil.which 함수는 주어진 실행 파일의 전체 경로를 반환합니다.
    # 만약 해당 실행 파일이 없거나 호출할 수 없다면 None을 반환합니다.
    if shutil.which(tesseract_cmd) is None:
        print("Tesserect is not installed.")
        return False
    else:
        print("Tesserect is installed.")
        return True

def create_zip_backup(src_folder, backup_folder):
    backup_folder = os.path.join(program_folder, "백업", backup_folder)
    file_list = os.listdir(os.path.join(program_folder, src_folder))
    filename = file_list[0]
    file_name = os.path.splitext(filename)[0]

    # src_folder의 이름만 가져옵니다.
    base_name = file_name
    
    # 백업 폴더가 없다면 생성합니다.
    # if not os.path.exists(backup_folder):
    #     os.makedirs(backup_folder)

    # if os.path.exists(os.path.join(backup_folder, f"{base_name}.zip")):
    #     base_name = os.path.splitext(base_name)[0]

    # zip 파일의 전체 경로를 만듭니다.
    archive_name = f"{base_name}.zip"
    archive_path = os.path.join(backup_folder, archive_name)

    # src_folder를 zip 파일로 압축합니다.
    shutil.make_archive(base_name, 'zip', src_folder)
    
    # 생성된 zip 파일을 백업 폴더로 이동시킵니다. 
    shutil.move(f"{base_name}.zip", archive_path)

def backup_file_move(src_folder, backup_folder):
    backup_folder = os.path.join("백업", backup_folder)
    file_list = os.listdir(src_folder)
    time.sleep(1)
    message["text"] = "기존 파일 백업 중"
    time.sleep(1)
    aver_time = []
    for index, filename in enumerate(file_list):
        start_time = time.time()
        total_files = len(file_list)
        src_file_path = src_folder + os.sep + filename
        shutil.copy(src_file_path, backup_folder)
        end_time = time.time()


        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files * 2
        message["text"] = f"백업 대상 파일 이동 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set((index + 1) / total_files * 100)
        root.update_idletasks()

def file_move(src_folder, backup_folder):
    # backup_folder = os.path.join("백업", backup_folder)
    file_list = os.listdir(src_folder)
    time.sleep(1)
    message["text"] = "파일 이동중"
    time.sleep(1)
    aver_time = []
    for index, filename in enumerate(file_list):
        start_time = time.time()
        total_files = len(file_list)
        src_file_path = src_folder + os.sep + filename
        shutil.copy(src_file_path, backup_folder)
        end_time = time.time()


        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        
        remaning_files = total_files - (index + 1)
        estimated_time = aver_copy_time * remaning_files * 2
        message["text"] = f"파일 이동 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set((index + 1) / total_files * 100)
        root.update_idletasks()
    message["text"] = "파일 이동 완료"
    time.sleep(2)
    message["text"] = "대기 중"

def file_backup(src, name):
     if os.path.exists(os.path.join(program_folder, src)):
        file_list = os.listdir(src)
        total_files = len(file_list)
        if len(file_list) > 0:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
            folder_name = f"{name}{formatted_datetime}"
            if not os.path.exists(os.path.join("백업", formatted_datetime)):
                message["text"] = "백업 폴더 생성 중"
                time.sleep(0.1)
                zip_folder_name = f"zip - {folder_name}"
                os.makedirs(os.path.join("백업", folder_name))
                message["text"] = "백업 대상 파일 이동 중"
                time.sleep(0.1)
                # os.makedirs(os.path.join("백업", zip_folder_name))
                # thread = threading.Thread(target=create_zip_backup, name="zip thread", args=(src, zip_folder_name))
                # thread.start()
                backup_file_move(src, folder_name)
                message["text"] = "백업 대상 파일 이동 완료"
                time.sleep(0.1)
                message["text"] = "기존 파일 삭제 중"
                time.sleep(0.1)

                aver_time = []
                
                for index, filename in enumerate(file_list):
                    start_time = time.time()
                    # time.sleep(0.5)
                    del_file = os.path.join(src, filename)
                    os.remove(del_file)
                    end_time = time.time()

                    aver_time.append(end_time - start_time)
                    aver_copy_time = sum(aver_time) / len(aver_time)
        
                    remaning_files = total_files - (index + 1)
                    estimated_time = aver_copy_time * remaning_files * 2
                    message["text"] = f"기존 파일 삭제 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
                    progress_var.set((index + 1) / total_files * 100)
                    root.update_idletasks()
                time.sleep(1)
                message["text"] = "기존 파일 삭제 완료"
            else:
                i = 0
                message["text"] = "중복 백업 폴더 감지 중"
                time.sleep(1)
                while os.path.exists(os.path.join("백업", formatted_datetime)):
                    i += 1
                formatted_datetime = f"{formatted_datetime} ({i})"
                message["text"] = "백업 폴더 생성 중"
                os.makedirs(formatted_datetime)
                message["text"] = "백업 대상 파일 압축 중"
                backup_file_move(src, formatted_datetime)
                message["text"] = "백업 대상 파일 압축 완료"
                message["text"] = "기존 파일 삭제 중"
                for index, filename in enumerate(file_list):
                    del_file = os.path.join(program_folder, src, filename)
                    os.remove(del_file)
                    progress_var.set((index + 1) / total_files * 200)
                    root.update_idletasks()
                    time.sleep(1)
                message["text"] = "기존 파일 삭제 완료"



def error_popup(title_message, message):
    error_window = tk.Tk()

    error_window.title("⚠️경고 : {}".format(title_message))
    # error_window.geometry("320x240+100+100")
    error_window.resizable(False, False)
    
    msg = tk.Message(error_window, text="{}".format(message), width=500, relief="flat")
    msg.pack()

    error_window.mainloop()

def pdf_file_move_check_popup(title_message, message, src, dest_folder):
    check_windows = tk.Tk()

    check_windows.title(f"{title_message}")
    check_windows.resizable(False, False)

    file_move_message = tk.Message(check_windows, text=f"{message}", width=500, relief="flat")
    file_move_message.pack()

    true_button = tk.Button(check_windows, text="네", command=lambda:file_move_thread(src, dest_folder), width=10, height=1)
    true_button.pack(side="left", padx=(10, 10))

    false_button = tk.Button(check_windows, text="아니오", command=check_windows.destroy, width=10, height=1)
    false_button.pack(side="right", padx=(10, 10))

    exit_button = tk.Button(check_windows, text="종료", command=check_windows.destroy, width=10, height=1)
    exit_button.pack(side="bottom", padx=(10, 10))

    check_windows.mainloop()


def file_move_thread(src, dest_folder):
    file_list = os.listdir(src)

    print(src, dest_folder)

    if len(file_list) > 0:
        backup_thread = threading.Thread(target=file_backup, name="file_backup_thread", args=(dest_folder, "원본 - "))
        backup_thread.start()
        backup_thread.join()

    thread = threading.Thread(target=file_move, name="file_move_thread", args=(src, dest_folder))
    thread.start()


def odd_number(): # 저장할 대상폴더 존재 여부 확인 및 대상 폴더 생성, 이름 변경 메서드 호출
    if not os.path.exists(os.path.join(program_folder, "앞면")):
        os.makedirs(os.path.join(program_folder, "앞면"))
    rename_files(1, "앞면")


def even_number(): # 저장할 대상폴더 존재 여부 확인 및 대상 폴더 생성, 이름 변경 메서드 호출
    if message["text"] == "대기 중":
        if not os.path.exists(os.path.join(program_folder, "뒷면")):
            os.makedirs(os.path.join(program_folder, "뒷면"))
        rename_files_reverse(2, "뒷면")

def auto_num():
    if message["text"] == "대기 중":
        if not os.path.exists(os.path.join(program_folder, "양면")):
            message["text"] = "출력 폴더 생성 중"
            os.makedirs(os.path.join(program_folder, "양면"))
        if not os.path.exists(os.path.join(program_folder, "process")):
            message["text"] = "작업 폴더 생성 중"
            os.makedirs(os.path.join(program_folder, "process"))
        txt = text.get()
        file_list = os.listdir(os.path.join(program_folder, "양면"))
        message["text"] = "백업 작업 중"
        backup_thread = threading.Thread(target=file_backup, name="backup_thread", args=(os.path.join(program_folder, "양면"), ''))
        backup_thread.start()
        backup_thread.join()
        # file_backup(os.path.join(program_folder, "양면"), '')
        message["text"] = "작업 중"
        thread = threading.Thread(target=automatic_Sorting, name="auto_thread", args=(1, 2, "양면", "양면", txt))
        thread.start()
        thread.join()
        # automatic_Sorting(1, 2, "양면", "양면", txt)
        message["text"] = "작업 완료"
        message["text"] = "대기 중"

def not_auto_num():
    if message["text"] == "대기 중":
        if not os.path.exists(os.path.join(program_folder, "양면")):
            message["text"] = "출력 폴더 생성 중"
            os.makedirs(os.path.join(program_folder, "양면"))
        if not os.path.exists(os.path.join(program_folder, "process")):
            message["text"] = "작업 폴더 생성 중"
            os.makedirs(os.path.join(program_folder, "process"))
        txt = text.get()
        file_list = os.listdir(os.path.join(program_folder, "양면"))
        message["text"] = "백업 작업 중"
        backup_thread = threading.Thread(target=file_backup, name="backup_thread", args=(os.path.join(program_folder, "양면"), ''))
        backup_thread.start()
        backup_thread.join()
        # file_backup(os.path.join(program_folder, "양면"), '')
        message["text"] = "작업 중"
        thread = threading.Thread(target=not_sort_automatic_Sorting, name="auto_thread", args=(1, 2, "양면", "양면", txt))
        thread.start()
        thread.join()
        # automatic_Sorting(1, 2, "양면", "양면", txt)
        message["text"] = "작업 완료"
        message["text"] = "대기 중"

def pdf_to_png(pdf_file, output_folder):
    file_list = os.listdir(os.path.join(program_folder, output_folder))
    # Open the PDF file
    pdf_document = fitz.open(pdf_file)
    aver_time = []
    pdf_page_count = int(pdf_document.page_count)

    if len(file_list) > 0:
        file_backup(output_folder, "PDF 변환 - ")        
    
    for page_number in range(int(pdf_document.page_count)):
        start_time = time.time()

        # Get the page
        page = pdf_document[page_number]

        image = page.get_pixmap()
        
        # Save the image as a PNG file
        image.save(f"{output_folder}/page_{page_number + 1}.png", "png")

        end_time = time.time()
        aver_time.append(end_time - start_time)
        aver_copy_time = sum(aver_time) / len(aver_time)
        percent = round((page_number + 1) / pdf_page_count * 100, 0)

        remaning_files = pdf_page_count - (page_number + 1)
        estimated_time = aver_copy_time * remaning_files * 2
        message["text"] = f"PDF 변환 중 {percent}%\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
        progress_var.set((page_number + 1) / pdf_page_count * 100)
        root.update_idletasks()

    # Close the PDF file
    pdf_document.close()

    message["text"] = "PDF 변환 완료"
    time.sleep(2)
    pdf_file_move_check_popup("파일 이동", "원본 폴더로 파일 이동을 하시겠습니까?",output_folder, os.path.join(program_folder, "원본"))
    message["text"] = "대기 중"

    # pdf_page_count = pdf_document.page_count()
    
    # for index, filename in enumerate(pdf_page_count):
    #     start_time = time.time()
    #     # time.sleep(0.5)
        
    #     end_time = time.time()

    #     aver_time.append(end_time - start_time)
    #     aver_copy_time = sum(aver_time) / len(aver_time)

    #     remaning_files = pdf_page_count - (index + 1)
    #     estimated_time = aver_copy_time * remaning_files * 2
    #     message["text"] = f"PDF 변환 중\n 예상 시간 : {estimated_time // 60:.0f}분 {estimated_time % 60:.0f}초"
    #     progress_var.set((index + 1) / pdf_page_count * 100)
    #     root.update_idletasks()

def auto_num_call():
    if message["text"] == "대기 중":
        if auto["auto"] == 'true':
            thread = threading.Thread(target=auto_num, name="auto_num_thread")
            thread.start()
        # thread.join()
        elif auto['auto'] == 'false':
            thread = threading.Thread(target=not_auto_num, name="not_auto_num_thread")
            thread.start()

def not_auto_num_call():
    if message["text"] == "대기 중":
        thread = threading.Thread(target=not_auto_num, name="not_auto_num_thread")
        thread.start()
        # thread.join()


def pdf_to_png_call():
    if message["text"] == "대기 중":
        message["text"] = "PDF 변환 중"
        time.sleep(2)
        file_list = os.listdir(os.path.join(program_folder, pdf_source))
        pdf_dest_folder = "PDF to PNG"
        # pdf_file = os.path.join(pdf_source, i)
        if not os.path.exists(os.path.join(program_folder, pdf_dest_folder)):
            os.makedirs(os.path.join(program_folder, pdf_dest_folder))
        if len(file_list) == 1:
            try:        
                for i in file_list:
                    thread = threading.Thread(target=pdf_to_png, name="thread", args=(os.path.join(pdf_source, i), pdf_dest_folder))
                    thread.start()
                # pdf_document = fitz.open(pdf_file)
                # pdf_page_count = pdf_document.page_count
                # for i in range(pdf_page_count):
                #     page = pdf_document[i]

               
            except:
                error_popup("원본 파일 오류", f"❗ PDF 원본 폴더에 파일이 존재하지 않습니다. \n   {pdf_source} 경로에 PDF 변환 대상 파일을 넣어주십시오.")
            message["text"] = "PDF 변환 완료"
            time.sleep(1)
            message["text"] = "대기 중"
        elif len(file_list) > 1:
            error_popup("원본 파일 오류", f"❗ PDF 원본 폴더에 파일이 {len(file_list)}개 있습니다.\n파일 개수를 1개로 줄여주세요.")

def pdf_to_png_call_thread():
    pdf_thread = threading.Thread(target=pdf_to_png_call, name="pdf_to_png_call")
    pdf_thread.start()
        
# def multi_page_pdf_trance_call():
#     error_popup("Multi Page converter", "아직 준비되지 않은 기능입니다.")

def update_message(*args):
    name_message["text"] = txt_var.get() + "1.파일확장자"

def text_search():
    ts = tk.Tk()
    ts.title("문서 내용 검색")
    ts.resizable(False, False)
    frame_top = tk.Frame(ts)
    frame_top.pack(side="top", pady=(20, 10))

    # search_txt_var = tk.StringVar()
    # search_txt_var.trace_add("write", update_message)

    search_text = tk.Entry(ts, width=40)
    search_text.pack(side="bottom")

    result_search_text = search_text.get()

    add_search_Button = tk.Button(ts, text="검색시작", command=lambda:button_search(search_text))
    print(search_text)
    add_search_Button.pack(side="right")

    search_message = tk.Label(ts, text="아래의 칸에 검색할 내용을 입력해주십시오. \n \n주의 : 검색어가 많을 경우 지연이 발생할 수 있습니다.")
    search_message.pack(side="top")

    s_progress_var = tk.DoubleVar()
    s_progress_bar = ttk.Progressbar(ts, variable=s_progress_var, length=300) # 작업 진행률 표기 막대그래프 정의
    s_progress_bar.pack(side="top")
    
    ts.mainloop()



def button_search(search_text):
    result_search_text = search_text.get()
    print("search text = " + result_search_text)
    i = search(result_search_text)
    
    # thread = threading.Thread(target=search, name="thread", args=(result_search_text))
    # thread.start()
    # i = thread.join()
    print("i : {}".format(i))
    search_result(i)
    


def search_result(result):
    sr = tk.Tk()
    sr.title("검색 결과")
    # sr.resizable(False, False)

    message = ""

    for i in range(len(result)):
        message += result[i]
        message += ", "

    msg = tk.Message(sr, text="검색결과 아래의 페이지에서 단어가 검색되었습니다. \n \n{}".format(message), width=500, relief="flat")
    msg.pack()

    sr.mainloop()

def engine_select(engine_sel):
    if engine_sel == 'tesseract':
        engine['engine'] = 'Tesseract'
        # print(engine)
        # return engine
    elif engine_sel == 'easyocr':
        engine["engine"] = 'EasyOCR'
        # print(engine)
        # return engine

def sort_select(sort_sel):
    if sort_sel == 'auto':
        auto["auto"] == 'true'
    elif sort_sel == 'not_auto':
        auto['auto'] == 'false'

def sys_info():
    info = tk.Tk()
    info.title("Software Info")
    info.resizable(False, False)
    frame_top = tk.Frame(info)
    frame_top.pack(side="top", padx=(20, 10), pady=(10, 5))

    version_var= tk.Label(info, text=f"\t{version}\t\nGPU 가속 가능 여부 : {torch.cuda.is_available()}\nTesseract 버전 : {pytesseract.get_tesseract_version()}\n업데이트 내역 : EasyOCR 엔진 사용시\n이미지 전처리 추가")
    version_var.pack(side="top", pady=(20, 10))
    # cuda_var = tk.Label(info, text=f"GPU 가속 가능 여부 : {torch.cuda.is_available()}")
    # cuda_var.pack(side="top", pady=(20, 10))
    # tesseract_var = tk.Label(info, text=f"Tessract 버전 : {pytesseract.get_tesseract_version()}")
    # tesseract_var.pack(side="top", pady=(20, 10))

    exit_button = tk.Button(info, text='종료', command=info.destroy, width=15, height=2)
    exit_button.pack(side="top", pady=(20, 10))

    info.mainloop()

def auto_popup():
    at = tk.Tk()
    at.title("자동 정렬")
    at.resizable(False, False)
    frame_top = tk.Frame(at)
    frame_top.pack(side="top", pady = (20, 10))
    
    auto_button = tk.Button(frame_top, text="양면", command=auto_num_call, width=15, height=2)
    auto_button.pack(side="right", padx=(10, 10))
   
    even_button = tk.Button(frame_top, text="뒷면", command=even_number, width=15, height=2) # Gui 버튼 정의
    even_button.pack(side="right", padx=(10, 10))

    stock_var = tk.StringVar()
    btn_stock1 = tk.Radiobutton(at, text="자동정렬", value="easyocr", variable=stock_var, command=lambda: sort_select('true'))
    btn_stock1.select()
    btn_stock2 = tk.Radiobutton(at, text="정렬", value="tesseract", variable=stock_var, command=lambda: sort_select('not_auto'))
    

    btn_stock1.pack()
    btn_stock2.pack()


def document_popup():
    dc = tk.Tk()
    dc.title("문서 변환")
    dc.resizable(False, False)
    frame_top = tk.Frame(dc)
    frame_top.pack(side="top", pady=(20, 10))

    # def change() :
    #     label1.config(text=stock_var.get()+"를 선택하셨습니다.")
    
    stock_var = tk.StringVar()
    btn_stock1 = tk.Radiobutton(dc, text="EasyOCR", value="easyocr", variable=stock_var, command=lambda: engine_select('easyocr'))
    btn_stock1.select()
    btn_stock2 = tk.Radiobutton(dc, text="Tesseract", value="tesseract", variable=stock_var, command=lambda: engine_select('tesseract'))
    

    btn_stock1.pack()
    btn_stock2.pack()
    
    ocr_button = tk.Button(frame_top, text="문자 변환", command=lambda: ocr_call(engine["engine"]), width=15, height=2)
    ocr_button.pack(side="left", padx=(10, 10))

    search_button = tk.Button(frame_top, text='검색', command=text_search, width=15, height=2)
    search_button.pack(side="left", padx=(10, 10))

    pdf_to_png_button = tk.Button(frame_top, text="PDF 사진 변환", command=pdf_to_png_call_thread, width=15, height=2)
    pdf_to_png_button.pack(side="bottom", padx=(10, 10))

    # multi_page_pdf_converter_button = tk.Button(frame_top, text="복수 페이지 변환", command=multi_page_pdf_trance_call, width=15, height=2)
    # multi_page_pdf_converter_button.pack(side="right", padx=(10, 10))

    dc.mainloop()


# 폴더 경로 설정
source_folder = "원본"  # 원본 폴더 이름
ocr_source_folder = "OCR_원본"
pdf_source = "PDF 변환"
program_folder = os.getcwd()  # 프로그램 폴더 경로

def ocr_call(engine):
    if message["text"] == "대기 중":
        backup_thread = threading.Thread(target=ocr, name="backup_thread", args=(os.path.join(ocr_source_folder), engine))
        backup_thread.start()
        # ocr(src_folder=os.path.join(ocr_source_folder), engine=engine)
    # print(engine)
    # ocr("원본")

# 필요한 폴더가 없으면 생성
if not os.path.exists(os.path.join(program_folder, source_folder)):
    os.makedirs(os.path.join(program_folder, source_folder))
if not os.path.exists(os.path.join(program_folder, ocr_source_folder)):
    os.makedirs(os.path.join(program_folder, ocr_source_folder))
if not os.path.exists(os.path.join(program_folder, pdf_source)):
    os.makedirs(os.path.join(program_folder, pdf_source))

# tkinter UI 생성
root = tk.Tk()
root.title(version)
# kwong daeshin darg

root.resizable(False, False)

frame_top = tk.Frame(root)
frame_top.pack(side="top", pady=(20, 10))

txt_var = tk.StringVar()
txt_var.trace_add("write", update_message)

text = tk.Entry(width=40, textvariable=txt_var)
text.pack(side="bottom")

name_message = tk.Label(root, text="1.파일확장자")
name_message.pack(side="bottom")

# text = tk.Text(width=40, height=1)
# text.pack(side="bottom")

# message = tk.Message(width=100, text=txt + "1.파일확장자")
# message.pack(side="bottom")

odd_button = tk.Button(frame_top, text="문자변환", command=document_popup, width=15, height=2) # Gui 버튼 정의
odd_button.pack(side="left", padx=(10, 10), pady=(1, 1))

# even_button = tk.Button(frame_top, text="뒷면", command=even_number, width=15, height=2) # Gui 버튼 정의
# even_button.pack(side="right", padx=(10, 10))

even_button = tk.Button(frame_top, text="정보", command=sys_info, width=15, height=2) # Gui 버튼 정의
even_button.pack(side="right", padx=(10, 10))

auto_button = tk.Button(frame_top, text="자동 정렬", command=auto_popup, width=15, height=2)
auto_button.pack(side="right", padx=(10, 10))

# reverse_button = tk.Button(frame_top, text="원터치", command=one_touch(), width=15, height=2)
# reverse_button.pack(side="bottom", padx=(10, 10))

frame_bottom = tk.Frame(root)
frame_bottom.pack(side="bottom", pady=(10, 20))

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(frame_bottom, variable=progress_var, length=300) # 작업 진행률 표기 막대그래프 정의
progress_bar.pack(side="top")

message = tk.Label(root, text="대기 중")
message.pack(side="bottom")

root.mainloop()