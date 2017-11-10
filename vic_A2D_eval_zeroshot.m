function vic_A2D_eval_zeroshot()

% -------------------------------------------------------------------------
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% email: vicky.kalogeiton@gmail.com

% If you use this software please cite our ICCV 2017 paper: 
% Joint learning of object and action detectors
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% In ICCV 2017

% -------------------------------------------------------------------------
% Function that computes the mAP for the zero-shot learning
% At training time we remove all actions of each object class and predict them 
% at test time
% -------------------------------------------------------------------------

if(~isdeployed), dbstop if error; end

% paths.test_detections: the path where the detections are stored 
paths.test_detections = [pwd '/'];

% Options for A2D 
options = [];
options = vic_options_A2D(options);

% Ground truth annotations for the A2D dataset
load('gt_test_A2D.mat','gt_test')

% Compute detections and AP 
% for each object class that we zero-shot its actions
ComputeBoxesFunction = str2func(['vic_compute_zeroshotBoxes']);
FinalAP = zeros(options.c_obj, options.c_act);
C = 0;
for cls_zeroshot = 1:options.c_obj
    % Load all detections in the format of a cell array (NxC), 
    % where C is the number of object-action classes and N is the number of frames
    % Each cell is a (Kx5) single matrix, where K is the number of detections
    % of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]
    % In A2D: (C = c_obj x c_act = 7 x 9) C = 63 and N = 2365
    [det_boxes] = ComputeBoxesFunction(gt_test.images, cls_zeroshot, paths, options);  
    for cls_act = 1:options.c_act
        C = C + 1;
        classname = [options.objects{cls_zeroshot} '_' options.actions{cls_act}];
        [res] = vic_map_objects_actions(gt_test, C, det_boxes, C);
        printAP = res.ap * 100;
        disp(['AP for ' classname ' is ' num2str(printAP) '%'])
        FinalAP(cls_act, cls_zeroshot) = printAP;
    end
end

% we measure AP over the classes that exist in the train/test sets
mAP = sum(FinalAP(find(FinalAP>0)))/options.num_valid; 


end

