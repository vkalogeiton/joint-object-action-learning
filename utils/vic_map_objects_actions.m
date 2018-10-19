function [res] = vic_map_objects_actions(gt_test, cls, det_boxes, cls_V)

%--------------------------------------------------------------------------
% det_boxes: multitask detections: cell array (CxN), where C is the number
% of object-action classes and N is the number of frames
% In A2D: (C = c_obj x c_act = 7 x 9) C = 63 and N = 2365
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

% for all frames
for k=1:size(det_boxes,2)
   disp([num2str(k) '/' num2str(size(det_boxes,2))])
   currBoxes = det_boxes{cls_V,k};
   pick = nms(currBoxes, 0.3);
   currBoxes = currBoxes(pick,:);
   idxCLSgt = find(~cellfun(@isempty,gt_test.boxes(k,:)));
   currentGT = [];
   if ~isempty(find(idxCLSgt == cls))
      currentGT = gt_test.boxes{k, idxCLSgt(find(idxCLSgt == cls))};
   end
   %if (idxCLSgt == cls), currentGT = gt_test.boxes{k, idxCLSgt}; end 
   %keyboard;
   if isempty(currentGT)
      ov = -1*ones(size(currBoxes,1),1);
   else
      try
          ov = computeOverlapTableDouble(double(currBoxes(:, 1:4)), currentGT);
      catch
          ov = vic_computeOverlapTableDouble(double(currBoxes(:, 1:4)), currentGT(:,1:4));
      end
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
    npos = npos + size(gt_test.boxes{ii,cls},1);
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

function [ov] = vic_computeOverlapTableDouble(boxes1, boxes2)

ov = zeros(size(boxes1,1), size(boxes2,1));
for ii=1:size(boxes1,1)
    bb1=boxes1(ii,:);
    for jj=1:size(boxes2,1)
        bb2=boxes2(jj,:);
        bi=[max(bb1(1),bb2(1)) ; max(bb1(2),bb2(2)) ; min(bb1(3),bb2(3)) ; min(bb1(4),bb2(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0
            % compute overlap as area of intersection / area of union
            ua=(bb1(3)-bb1(1)+1)*(bb1(4)-bb1(2)+1)+...
              (bb2(3)-bb2(1)+1)*(bb2(4)-bb2(2)+1)-...
              iw*ih;
            ov(ii,jj)=iw*ih/ua;
       end
    end
end

end

