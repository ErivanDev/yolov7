import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def calculate_iou(boxA, boxB):
    # boxA and boxB should be in the format (x1, y1, x2, y2)
    # Intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Intersection area
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Area of each box
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Union area
    union_area = boxA_area + boxB_area - inter_area

    # Calculate IoU
    iou = inter_area / float(union_area)

    return iou

import numpy

def detect(save_img=False):
    toExport = False

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    contador = 0
    w_max, h_max = 0, 0
    images0 = []
    questions = {}
    markers = {}
    omarkers = {}
    q_location = []
    lps = 12

    if toExport == False:
        import json
        file_path = "/content/sample_data/markers.json"
        with open(file_path, "r") as file_json:
            omarkers = json.load(file_json)

    # Exibir a lista de dicionários carregada
    # print(omarkers)
    file_path = ''

    for path, img, im0s, vid_cap in dataset:
        file_path = path

        # cv2.imwrite('/content/img.jpg', img)
        # cv2.imwrite('/content/im0s.jpg', img)

        # print(img.shape)
        # print(im0s.shape)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2GRAY)

        # img = cv2.rotate(img, cv2.ROTATE_180)
        # im0s = cv2.rotate(im0s, cv2.ROTATE_180)
        angle = 0 # 180

        image_center = tuple(numpy.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        image_center = tuple(numpy.array(im0s.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # im0s = cv2.warpAffine(im0s, rot_mat, im0s.shape[1::-1], flags=cv2.INTER_LINEAR)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        print('img', img.shape)
        # img torch.Size([1, 3, 640, 480])

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        order1 = []
        order2 = []
        
        temp1 = []
        temp2 = []
        
        mark1 = []
        mark2 = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    h = int(xyxy[3]) - int(xyxy[1])
                    w = int(xyxy[2]) - int(xyxy[0])

                    if w > w_max:
                        w_max = w
                    if h > h_max:
                        h_max = h

                    # print(img.shape)
                    temp = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    
                    # temps.append(temp)
                    # images0.append(temp)
                    
                    ratio = img.shape[2]/img.shape[3]

                    # print('ratio', ratio)

                    # q_location.append((int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])))

                    # A3:
                    if ratio > 0.60 and ratio < 0.80: 
                        if((int(xyxy[0]) < 100 and contador%2==1) or (int(xyxy[0]) > 900 and contador%2==0)):
                            order1.append(int(xyxy[1]))
                            temp1.append(temp)
                            mark1.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))
                        else:
                            order2.append(int(xyxy[1]))
                            temp2.append(temp)
                            mark2.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))

                    # A4:
                    if ratio > 1.08 and ratio < 1.72:
                        order1.append(int(xyxy[1]))
                        temp1.append(temp)
                        mark1.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))

            temp1 = sorted(zip(order1, temp1))
            temp1 = [x[1] for x in temp1]
            
            mark1 = sorted(zip(order1, mark1))
            mark1 = [x[1] for x in mark1]

            temp2 = sorted(zip(order2, temp2))
            temp2 = [x[1] for x in temp2]
            
            mark2 = sorted(zip(order2, mark2))
            mark2 = [x[1] for x in mark2]

            i = 0
            for t, m in zip(temp1, mark1):
                if contador + i/10 not in []:
                    questions[contador + i/10] = t
                    markers[contador + i/10] = m

                i+=1
            
            i = 0
            for t, m in zip(temp2, mark2):
                if ((lps*2)-contador-1) + i/10 not in []:
                    questions[((lps*2)-contador-1) + i/10] = t
                    markers[((lps*2)-contador-1) + i/10] = m

                i+=1
            
            # print(questions)

            # print(len(det))

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        contador+=1

    markers = sorted(zip(markers.keys(), markers.values()))
    # print('markers keys', markers)
    
    if toExport == True:
        import json
        print('markers', markers)
        with open('/content/sample_data/markers.json', "w") as file_json:
            json.dump(markers, file_json)
    
    # mks = [x[0] for x in markers]
    # markers = [x[1] for x in markers]

    confiance = 0
    # while(len(markers) < len(omarkers)):
    #     markers.append((int(xyxy[1]),int(xyxy[3]),int(xyxy[0]),int(xyxy[2])))

    print('markers', len(markers), len(omarkers))

    different = abs(len(markers) - len(omarkers))

    # import csv
    # with open('/content/sample_data/confiance.csv', "a", newline="") as arquivo_csv:
    #     escritor_csv = csv.DictWriter(arquivo_csv, fieldnames=["path", "valid"])
    #     escritor_csv.writerow(
    #         {
    #             'path': file_path.split('/')[-2],
    #             'valid': invalid,
    #         }
    #     )

    vetor_ordenado = sorted(zip(questions.keys(), questions.values()))
    images0 = [x[1] for x in vetor_ordenado]

    check = True
    stop = 0
    while(check):
        invalid = 0

        print(markers)
        print('------------------------------')
        
        for (mk, omk) in zip(markers, omarkers):
            print('keys', mk[0], omk[0])
            print('iou', calculate_iou(mk[1], omk[1]))
            
            if mk[0]//1 != omk[0]//1:
                invalid += 1

                index = markers.index(mk)
                del markers[index]
                del images0[index]

                break

            if calculate_iou(mk[1], omk[1]) <= 0.6:
                invalid += 1

                index = markers.index(mk)
                del markers[index]
                del images0[index]
                
                # markers.remove(mk)
                # images0.remove(i)

                break

        # print(invalid, stop)
        if invalid == 0:
            check = False

        '''
        if invalid == 0 and len(markers) == len(omarkers):
            check = False

        if len(markers) < len(omarkers):
            check = False

        stop += 1
        if stop > 1000:
            check = False
        '''

    print('markers', len(markers), len(omarkers))

    if len(markers) < len(omarkers):
        invalid += 1

    temp = numpy.ones([ h_max*13, w_max*4, 3 ]) * 255

    # print('lenght', len(images0))

    # print('images0', images0[0].shape)
    
    for wi in range(4):
        for hi in range(13):
            if (hi*4+wi < len(images0)):
                im1 = images0[hi*4+wi]
                temp[(h_max*hi):(h_max*hi)+im1.shape[0],
                     (w_max*wi):(w_max*wi)+im1.shape[1]] = im1

    # cv2.imwrite(source.replace('ProvasJPGs','ProvasGroup')+'.jpg', temp)
    
    import os
    if not os.path.exists('/'.join(source.replace('_aligned','_yolo').split('/')[:-1])):
        os.makedirs('/'.join(source.replace('_aligned','_yolo').split('/')[:-1]))
    
    cv2.imwrite(source.replace('_aligned','_yolo')+'.jpg', temp)
    # file_name = source.replace(
    #     '/content/drive/MyDrive/EduTech/Projetos/LeitorDeProvas/ProvasJPGs/JORGE-HERBERT-PORT-MAT-2023/MATEMATICA-45-CADERNO-01/',
    #      ''
    # )
    # print(file_name)

    if invalid == 0:
        import os
        import shutil

        if not os.path.exists('/'.join(source.replace('_aligned','_valid').split('/')[:-1])):
            os.makedirs('/'.join(source.replace('_aligned','_valid').split('/')[:-1]))
    
        o = source.replace('_aligned','_yolo') + '.jpg'
        d = source.replace('_aligned','_valid') + '.jpg'

        shutil.copy(o, d)
    else:
        import os
        import shutil

        if not os.path.exists('/'.join(source.replace('_aligned','_invalid').split('/')[:-1])):
            os.makedirs('/'.join(source.replace('_aligned','_invalid').split('/')[:-1]))

        o = source.replace('_aligned','_yolo') + '.jpg'
        d = source.replace('_aligned','_invalid') + '.jpg'

        shutil.copy(o, d)

    # cv2.imwrite('/content/sample_data/'+str(file_name)+'.jpg', temp)
    cv2.imwrite('/content/sample_data/test.jpg', temp)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    # print([x[0] for x in vetor_ordenado])

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
