function vic_A2D_eval(learning_case_wanted)

% function that computes the mAP for the multitask, hierarchical and
% cartesian combinations

if (nargin<1), learning_case_wanted = 2; end
if(~isdeployed), dbstop if error; end

% paths.test_detections: the path where the detections are stored 
paths.test_detections = [pwd '/'];

allcases = {'multitask', 'hierarchical', 'cartesian'};
options.learning_case = allcases{learning_case_wanted}; 

% Options for A2D 
options = vic_options_A2D(options);

% Ground truth annotations for the A2D dataset
load('gt_test_A2D.mat','gt_test')

% Detections

% Load detections for all boxes in the following format: cell array 
% (multitask): (NxC), where C is the number of object-action classes
% (cartesian): (NxV), where V is the number of valid object-action pairs
% (hierarchical): (NxV), where V is the number of valid object-action pairs
% and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]
% In A2D: (C = c_obj x c_act = 7 x 9) C = 63, V = 43 and N = 2365
ComputeBoxesFunction = str2func(['vic_compute_' options.learning_case 'Boxes']);
[det_boxes] = ComputeBoxesFunction(gt_test.images, paths, options);  

FinalAP = zeros(options.c_obj, options.c_act);
C = 0;
V = 0; 
switch options.learning_case
    case 'multitask'
        for cls_obj = 1:options.c_obj
            for cls_act = 1:options.c_act
                C = C + 1;
                V = C; 
                classname = [options.objects{cls_obj} '_' options.actions{cls_act}];
                [res] = vic_map_objects_actions(gt_test, C, det_boxes, V);
                printAP = res.ap * 100;
                disp(['AP for ' classname ' is ' num2str(printAP) '%'])
                FinalAP(cls_act, cls_obj) = printAP;
            end
        end
    case {'cartesian','hierarchical'}
        for cls_obj = 1:options.c_obj
            for cls_act = 1:options.c_act
                C = C + 1;
                if options.AllCombinations(C, 4) ~=0
                    V = V + 1; 
                    classname = [options.objects{cls_obj} '_' options.actions{cls_act}];
                    [res] = vic_map_objects_actions(gt_test, C, det_boxes, V);
                    printAP = res.ap * 100;
                    disp(['AP for ' classname ' is ' num2str(printAP) '%'])
                    FinalAP(cls_act, cls_obj) = printAP;
                end
            end
        end
end
% we measure AP over the classes that exist in the train/test sets
mAP = sum(FinalAP(find(FinalAP>0)))/options.num_valid; 


end

