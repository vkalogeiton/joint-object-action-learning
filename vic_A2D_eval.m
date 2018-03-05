function vic_A2D_eval(learning_case_wanted)

% -------------------------------------------------------------------------
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% email: vicky.kalogeiton@gmail.com

% If you use this software please cite our ICCV 2017 paper: 
% Joint learning of object and action detectors
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% In ICCV 2017

% -------------------------------------------------------------------------
% function that computes the mAP for the multitask, hierarchical and
% cartesian combinations
% -------------------------------------------------------------------------

if (nargin<1), learning_case_wanted = 1; end
if(~isdeployed), dbstop if error; end

addpath([pwd '/utils/'])
% paths.test_detections: the path where the detections are stored 
paths.test_detections = ['pwd' '/detections/'];
allcases = {'multitask', 'hierarchical', 'cartesian'};
options.learning_case = allcases{learning_case_wanted}; 

% Options for A2D 
options = vic_options_A2D(options);

% Ground truth annotations for the A2D dataset
load([pwd '/src/gt_test_A2D.mat'],'gt_test')

% Detections

% Load detections for all boxes in the following format: cell array 
% (multitask): (CxN), where C is the number of object-action classes
% (cartesian): (VxN), where V is the number of valid object-action pairs
% (hierarchical): (VxN), where V is the number of valid object-action pairs
% and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]
% In A2D: (C = c_obj x c_act = 7 x 9) C = 63, V = 43 and N = 2365

ComputeBoxesFunction = str2func(['vic_compute_' options.learning_case 'Boxes']);
[det_obj_boxes, det_act_boxes, det_obj_act_boxes] = ComputeBoxesFunction(gt_test.images, paths, options);  
        
FinalAP = zeros(options.c_obj, options.c_act);
FinalAP_O  = zeros(options.c_obj, 1);
FinalAP_A  = zeros(options.c_act, 1);
C = 0;
V = 0; 

switch options.learning_case
    case 'multitask'
        % object-action pairs
        for cls_obj = 1:options.c_obj
            for cls_act = 1:options.c_act
                C = C + 1;
                V = C; 
                classname = [options.objects{cls_obj} '_' options.actions{cls_act}];
                [res] = vic_map_objects_actions(gt_test, C, det_boxes, V);
                printAP = res.ap * 100;
                disp(['AP for ' classname ' is ' num2str(printAP) '%'])
                FinalAP(cls_obj, cls_act) = printAP;
            end
        end
        % only objects 
        for cls_obj = 1:options.c_obj
            classname = options.objects{cls_obj}; 
            [res_obj] = vic_map(gt_test, cls_obj, det_obj_boxes, options, true); 
            printAP = res_obj.ap * 100;
            disp(['AP for ' classname ' is ' num2str(printAP) '%'])
            FinalAP_O(cls_obj) = printAP;
        end
        % only actions
        for cls_act = 1:options.c_act
            classname = options.actions{cls_act}; 
            [res_act] = vic_map(gt_test, cls_act, det_act_boxes, options, false); 
            printAP = res_act.ap * 100;
            disp(['AP for ' classname ' is ' num2str(printAP) '%'])
            FinalAP_A(cls_act) = printAP;
        end
    case {'cartesian','hierarchical'}
        % object-action pairs
        for cls_obj = 1:options.c_obj
            for cls_act = 1:options.c_act
                C = C + 1;
                if options.AllCombinations(C, 4) ~=0
                    V = V + 1; 
                    classname = [options.objects{cls_obj} '_' options.actions{cls_act}];
                    [res] = vic_map_objects_actions(gt_test, C, det_boxes, V);
                    printAP = res.ap * 100;
                    disp(['AP for ' classname ' is ' num2str(printAP) '%'])
                    FinalAP(cls_obj, cls_act) = printAP;
                end
            end
        end
        % only objects 
        for cls_obj = 1:options.c_obj
            classname = options.objects{cls_obj}; 
            [res_obj] = vic_map(gt_test, cls_obj, det_obj_boxes, options, true); 
            printAP = res_obj.ap * 100;
            disp(['AP for ' classname ' is ' num2str(printAP) '%'])
            keyboard;
            FinalAP_O(cls_obj) = printAP;
        end
        % only actions
        for cls_act = 1:options.c_act
            classname = options.actions{cls_act}; 
            [res_act] = vic_map(gt_test, cls_act, det_act_boxes, options, false); 
            printAP = res_act.ap * 100;
            disp(['AP for ' classname ' is ' num2str(printAP) '%'])
            FinalAP_A(cls_act) = printAP;
        end
end

% we measure AP over the classes that exist in the train/test sets
mAP = sum(FinalAP(find(FinalAP>0)))/options.num_valid; 
disp(['mAP for ' options.learning_case 'on object-action pairs is ' num2str(mAP) '%'])

mAP_O = sum(FinalAP_O)/options.c_obj; 
disp(['mAP for ' options.learning_case 'on objects is ' num2str(mAP_O) '%'])

% 'none' class is also background/negative for actions, so when evaluating only actions we don't consider it
mAP_A = sum(FinalAP_A(2:end))/(options.c_act-1); 
disp(['mAP for ' options.learning_case 'on actions is ' num2str(mAP_A) '%'])
end

