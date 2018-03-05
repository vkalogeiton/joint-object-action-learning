function [res] = vic_map(gt_test, cls_X, det_boxes, options, objects)

%--------------------------------------------------------------------------
% det_boxes: multitask detections: cell array (XxN), where X is the number
% of object or action classes and N is the number of frames
% In A2D: X: c_obj = 7 or X: c_act = 9, and N = 2365
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]

% gt_test: struct with two fields
% -- path: (Nx1) cell array containing the frame names
% -- boxes:(NxC) cell array with the ground truth bounding boxes
% The cell {n,1} contains a Kx4 array, where K is the number of ground
% truth boxes of the frame GT_List.path{n,1} [x1, y1, x2, y2]
%--------------------------------------------------------------------------

if(~isdeployed)
    dbstop if error
end
       
AllLabels = cell(size(det_boxes,2),1);
AllScores = cell(size(det_boxes,2),1);

% objects
if (nargin<5), objects = true; end

if objects
  idx_gt = find(options.AllCombinations(:, 1) == cls_X); 
else % actions
  idx_gt = find(options.AllCombinations(:, 2) == cls_X); 
end

% for all frames
for k=1:size(det_boxes,2)
   disp([num2str(k) '/' num2str(size(det_boxes,2))])
   currBoxes = det_boxes{cls_X,k};
   pick = nms(currBoxes, 0.3);
   currBoxes = currBoxes(pick,:);
   idxCLSgt = find(~cellfun(@isempty, gt_test.boxes(k,:)));
   currentGT = [];
   for icls = 1:length(idxCLSgt)
    if ~isempty(find(idx_gt == idxCLSgt(icls)))
      currentGT = cat(1,currentGT, gt_test.boxes{k, idxCLSgt(icls)}); 
    end
   end
   %keyboard;
   if isempty(currentGT)
      ov = -1*ones(size(currBoxes,1),1);
   else
      ov = computeOverlapTableDouble(double(currBoxes(:, 1:4)), currentGT);
      ov = double(ov>=0.5);
      ov(ov==0) = -1;
            
      for bb=1:size(currentGT,1)
          idx = find(ov(:,bb)==1,1);
          ov(idx+1:end,bb) = -1;
      end
      ov = max(ov,[],2);
   end

   AllLabels{k,1} = ov;
   AllScores{k,1} = currBoxes(:,5);

end
AllLabels = cell2mat(AllLabels);
AllScores = cell2mat(AllScores);    
    
[aa,bb] = sort(AllScores,'descend');
tp = (AllLabels(bb)==1);
fp = (AllLabels(bb)==-1);
 % get npos
npos = 0;
for ii=1:size(gt_test.boxes,1)
    for icls = 1:length(idx_gt)
      npos = npos + size(gt_test.boxes{ii,idx_gt(icls)},1);
    end
end
    
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);
    
ap=0;
for t=0:0.1:1
  p=max(prec(rec>=t));
  if isempty(p)
     p=0;
  end
  ap=ap+p/11;
end

res.ap = ap;
res.precision = prec; 
res.recall = rec; 

end

