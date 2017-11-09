function options = vic_options_A2D(options)

% A2D has c_obj = 7 object classes and c_act = 9 action classes
% In total there are C = 7x9 = 63 combinations
% out of which (valid_pairs) V = 43 are valid (for instance there is no car
% eating)

% options.objects = {'adult', 'baby', 'ball', 'bird', 'car' , 'cat', 'dog'};
% options.actions = {'none','climbing', 'crawling', 'eating', 'flying', 'jumping', 'rolling', 'running', 'walking'};
% options.c_obj = length(options.objects); % number of object classes 
% options.c_act = length(options.actions); % number of action classes 

% options.actions_given_objects2{1,1} = {'none', 'climbing', 'crawling', 'eating', 'jumping',  'rolling', 'running', 'walking'};
% options.actions_given_objects2{2,1} = {'none', 'climbing', 'crawling', 'rolling', 'walking'};
% options.actions_given_objects2{3,1} = {'none', 'flying',   'jumping',  'rolling'};
% options.actions_given_objects2{4,1} = {'none', 'climbing', 'eating',   'flying',  'jumping', 'rolling', 'walking'};
% options.actions_given_objects2{5,1} = {'none', 'flying',   'jumping',  'rolling', 'running'};
% options.actions_given_objects2{6,1} = {'none', 'climbing', 'eating',   'jumping', 'rolling', 'running', 'walking'};
% options.actions_given_objects2{7,1} = {'none', 'crawling', 'eating',   'jumping', 'rolling', 'running', 'walking'};  
        
options.valid_pairs= {'adult_none', 'adult_climbing', 'adult_crawling', ...
    'adult_eating', 'adult_jumping', 'adult_rolling', 'adult_running', 'adult_walking', ...
    'baby_none', 'baby_climbing', 'baby_crawling', 'baby_rolling', 'baby_walking', ...
    'ball_none', 'ball_flying', 'ball_jumping','ball_rolling', ...
    'bird_none', 'bird_climbing', 'bird_eating', 'bird_flying', ...
    'bird_jumping', 'bird_rolling', 'bird_walking', ...
    'car_none', 'car_flying', 'car_jumping', 'car_rolling', 'car_running', ...
    'cat_none', 'cat_climbing', 'cat_eating', 'cat_jumping', ...
    'cat_rolling', 'cat_running', 'cat_walking', ...
    'dog_none', 'dog_crawling', 'dog_eating', 'dog_jumping',...
    'dog_rolling', 'dog_running', 'dog_walking'};     
options.num_valid = length(options.valid_pairs); 

objects = cell(1,1); 
actions = cell(1,1); 
for ii = 1:length(options.valid_pairs)
    names = [];
    names = strsplit(options.valid_pairs{1,ii},'_');
    if isempty(objects{1,1})
        objects{1,1} = names{1};
    elseif isempty(find(ismember(objects,names{1})))
        objects = [objects; names{1}];
    end
    if isempty(actions{1,1})
        actions{1,1} = names{2};
    elseif isempty(find(ismember(actions,names{2})))
        actions = [actions; names{2}];
    end    
end

AllCombinations = zeros(length(objects)*length(actions),4); 
% The first column has the object id 
% The second column has the action id 
% The third column has 1 if it's a valid combination, otherwise 0
% The last column has the object id of the valid combinations
C = 1; 
V = 1; 
for cls_obj = 1:length(objects)
   for cls_act = 1:length(actions)
       AllCombinations(C, 1) = cls_obj;
       AllCombinations(C, 2) = cls_act;
       candidate_cls = find(ismember(options.valid_pairs,[objects{cls_obj} '_' actions{cls_act}]));
       if ~isempty(candidate_cls)
           AllCombinations(C, 3) = 1; 
           AllCombinations(C, 4) = V;
           V = V + 1; 
       end
       C = C + 1;        
   end    
end
options.AllCombinations = AllCombinations; 
options.objects = objects;
options.actions = actions;
options.c_obj = length(options.objects); % number of object classes 
options.c_act = length(options.actions); % number of action classes 

% Find actions given objects
for cls_obj = 1:options.c_obj
    idx_act = find(AllCombinations(:,1) == cls_obj & AllCombinations(:,3) == 1);
    options.actions_given_objects{cls_obj,1} = actions(AllCombinations(idx_act(:),2));
end



end
