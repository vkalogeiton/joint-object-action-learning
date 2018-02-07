function [res] = vic_map_objects_actions(GT_List, cls_obj, cls_act, det_boxes, det_cls, obj_loc, act_loc)
%--------------------------------------------------------------------------
% det_boxes: multitask detections: cell array (CxN), where C is the number
% of object-action classes and N is the number of frames
% In A2D: (C = c_obj x c_act = 7 x 9) C = 63 and N = 2365
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]

% det_cls: is the object-action class to detect
% In A2D: det_cls = 1, ..., 63

% GT_List: struct with two fields
% -- path: (Nx1) cell array containing the frame names
% -- box:  (Nx1) cell array with the ground truth bounding boxes 
% The cell {n,1} contains a Kx6 array, where K is the number of ground
% truth boxes of the frame GT_List.path{n,1} and 6 are [bbox coordinates, cls_obj, cls_act]
% [x1, y1, x2, y2, cls_obj, cls_act]

% cls_obj, cls_act: the ground truth classes for objects and actions,
% respectively 
%--------------------------------------------------------------------------

if (nargin < 6), obj_loc = 5; end
if (nargin < 7), act_loc = 6; end

%--------------------------------------------------------------------------
AllLabels = cell(length(det_boxes), 1);
AllScores = cell(length(det_boxes), 1);

for k=1:size(det_boxes, 2)
   disp([num2str(k) '/' num2str(length(det_boxes))])
   currBoxes = det_boxes{det_cls ,k};
   pick = nms(currBoxes, 0.3);
   currBoxes = currBoxes(pick, :);

   idxCLSgt = find(GT_List.box{k}(:, obj_loc) == cls_obj & ...
       GT_List.box{k}(:, act_loc) == cls_act);
   currentGT = GT_List.box{k}(idxCLSgt, 1:4) + 1;

   if isempty(currentGT)
      ov = -1*ones(size(currBoxes, 1),1);
   else
      ov = computeOverlapTableDouble(double(currBoxes(:, 1:4)), currentGT);
      ov = double(ov >= 0.5);
      ov(ov == 0) = -1;
            
      for bb=1:size(currentGT, 1)
          idx = find(ov(:, bb) == 1, 1);
          ov(idx+1:end, bb) = -1;
      end
      ov = max(ov, [], 2);
   end

   AllLabels{k, 1} = ov;
   AllScores{k, 1} = currBoxes(:, 5);

end
AllLabels = cell2mat(AllLabels);
AllScores = cell2mat(AllScores);    
    
[~,bb] = sort(AllScores, 'descend');
tp = (AllLabels(bb) == 1);
fp = (AllLabels(bb) == -1);

% get npos
npos = 0;
for ii=1:length(GT_List.box)
    npos = npos + length(find(GT_List.box{ii}(:, obj_loc) == cls_obj & ...
        GT_List.box{ii}(:, act_loc)== cls_act ));
end

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);
    
ap=0;
for t=0:0.1:1
  p=max(prec(rec >= t));
  if isempty(p)
     p=0;
  end
  ap=ap+p/11;
end

res.ap = ap;
res.precision = prec; 
res.recall = rec; 

end
