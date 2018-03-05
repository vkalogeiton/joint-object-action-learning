function [det_obj_boxes, det_act_boxes, det_obj_act_boxes] = vic_compute_hierarchicalBoxes(GT_List, paths, options)

% -------------------------------------------------------------------------
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% email: vicky.kalogeiton@gmail.com

% If you use this software please cite our ICCV 2017 paper: 
% Joint learning of object and action detectors
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% In ICCV 2017

%--------------------------------------------------------------------------
% toy function that shows how to compute the detection boxes 
% for the hierarchical case 
%--------------------------------------------------------------------------
% Input 
%--------------------------------------------------------------------------
% paths.test_detections: the path where the detections are stored 

% GT_List:cell array (1xN) that contains the N ground truth frames 

% In A2D: c_obj = 7, c_act = 9, C = 63, V = 43 and N = 2365

if (nargin < 3) 
    options.c_obj = 7; % number of object classes
    options.c_act = 9; % number of action classes
    options.learning_case = 'hierarchical'; 
end

%--------------------------------------------------------------------------
% Output
%--------------------------------------------------------------------------
% det_obj_boxes: multitask detections for object-only evaluation: cell array (c_objxN), 
% where c_obj is the number of object classes and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]

% det_act_boxes: multitask detections for actions-only evaluation: cell array (c_actxN), 
% where c_act is the number of action classes and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]

% det_obj_act_boxes: multitask detections: cell array (VxN), where V is the number
% of valid object-action and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]
%--------------------------------------------------------------------------

n_frames = length(GT_List);  

% evaluate both actions and objects for the hierarchical case
for ii=1:n_frames
    det = [];
    det = load([paths.test_detections GT_List{ii} '_' options.learning_case '.mat']);  
    % we assume that the det is a struct with fields: 
    % -- boxes: Kx[(c_obj+1)x4]: [bbox_c_obj1, bbox_c_obj2, ..., bbox_c_obj]
    % the regression is done per object (the first one is for background)
    % -- score: Kx[(c_obj+1)] object scores
    % -- act_score Kx[(V)] action/given objects scores
    % we include all our detections

    % object-action pair 
    V = 0; 
    for cls_obj = 1:options.c_obj
        bboxes = [];
        clear starting ending
        starting = (cls_obj)*4 + 1; 
        ending = starting + 3; 
        bboxes = det.boxes(:, starting:ending); 
        bboxes = bboxes +1; % boxes from python to matlab format
        for cls_act = 1:length(options.actions_given_objects{cls_obj, 1})            
            V = V + 1;
            bboxes(:, 5) = det.score(:, cls_obj+1) .* det.act_score(:, V);
            bboxes = single(bboxes); 
            [~, ind] = sort(bboxes(:, 5), 'descend');
            det_obj_act_boxes{V, ii} = bboxes(ind, :);   
        end      
    end

    % only objects
    for cls_obj = 1:options.c_obj
        clear starting ending
        starting = (cls_obj)*4 + 1; 
        ending = starting + 3; 
        bboxes = [];
        bboxes = det.boxes(:, starting:ending); 
        bboxes = bboxes + 1; % boxes from python to matlab format
        bboxes(:, 5) = det.score(:, cls_obj+1);
        bboxes = single(bboxes); 
        [~, ind] = sort(bboxes(:, 5), 'descend');
        det_obj_boxes{cls_obj, ii} = bboxes(ind, :);   
    end 

    % only actions
    for cls_act=1:options.c_act
        all_idx_obj = find(options.AllCombinations(:, 2) == cls_act & options.AllCombinations(:, 3) == 1);
        all_objects_given_action = options.AllCombinations(all_idx_obj, 1); 
        all_cls_act_obj_V = options.AllCombinations(all_idx_obj, 4);
        scores = 0;
        tmp_scores = [];
        for idx_obj = 1:length(all_objects_given_action)
            cls_obj = all_objects_given_action(idx_obj); 
            cls_act_obj = all_cls_act_obj_V(idx_obj);
            % P(cls_obj) * P(action/object)
            tmp_scores(:, idx_obj) = det.score(:, cls_obj+1) .* det.act_score(:, cls_act_obj); 
            % Sum (P(cls_obj) * P(action/object)) over all cls_obj
            scores = scores + tmp_scores(:, idx_obj); 
        end
        
        max_scores = [];
        [~, max_scores] = max(tmp_scores,[],2);
        cls_obj_idx = [];
        cls_obj_idx = all_objects_given_action(max_scores');
        bboxes= []; 
        for kk=1:length(cls_obj_idx)
           starting = cls_obj_idx(kk)*4 + 1;
           ending = starting + 3;
           bboxes(kk, 1:4) =  det.boxes(kk, starting:ending);
        end
        bboxes(:, 5) = scores;
        bboxes = single(bboxes);
        [~, ind] = sort(bboxes(:, 5), 'descend');
        det_act_boxes{cls_act, ii} = bboxes(ind, :);
    end
end
        
end
