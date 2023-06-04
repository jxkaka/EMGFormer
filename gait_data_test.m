clc;
%% 
data = readtable('/Users/jiaxuan/Library/CloudStorage/Dropbox/WorkSpace/GaitData_process/WHY_20230211/normal/all_WYH_normal1.csv');
% data_index = readtable('/Users/chenzcha/Desktop/WorkSpace/GaitData_process/WHY_20230211/normal/all_WYH_normal1_index.csv');

dat_matrix = table2array(data);  
% dat_index = table2array(data_index); 

clear data data_index

% segmentation: check how many cell can be segmentated
sample_size = 1600;
num_samples = floor(size(dat_matrix,1) / sample_size);

sagment_matrix = cell(num_samples,1);
for i = 1:num_samples
    sample_start = (i-1)*sample_size + 1;
    sample_end = i*sample_size;
    sagment_matrix{i} = dat_matrix(sample_start:sample_end,:);
end

% delete the remaining rows
delete_data1 = dat_matrix(1:num_samples*sample_size,:);

clear i sample_size sample_start sample_end num_samples delete_data1

%% segmentation: previous100 + 1600 + next100

sample_size = 1600;
num_samples = floor(size(dat_matrix,1) / sample_size);

segmented_dda = cell(num_samples,1);
for i = 1:num_samples
    sample_start = (i-1)*sample_size + 1;
    sample_end = i*sample_size;
    
    % Check if it is the first sample
    if i == 1
        % Only add the next 100 rows
        current_sample = dat_matrix(sample_start:sample_end,:);
        next_sample_start = sample_end + 1;
        next_sample_end = sample_end + 200;
        next_sample = dat_matrix(next_sample_start:next_sample_end,:);
        current_sample = [current_sample; next_sample];
    % Check if it is the last sample
    elseif i == num_samples
        % Only add the previous 100 rows
        prev_sample_start = sample_start - 200;
        prev_sample_end = sample_start - 1;
        prev_sample = dat_matrix(prev_sample_start:prev_sample_end,:);
        current_sample = dat_matrix(sample_start:sample_end,:);
        current_sample = [prev_sample; current_sample];
    else
        % Add previous 100 rows and next 100 rows
        prev_sample_start = sample_start - 100;
        prev_sample_end = sample_start - 1;
        prev_sample = dat_matrix(prev_sample_start:prev_sample_end,:);
        current_sample = dat_matrix(sample_start:sample_end,:);
        next_sample_start = sample_end + 1;
        next_sample_end = sample_end + 100;
        next_sample = dat_matrix(next_sample_start:next_sample_end,:);
        current_sample = [prev_sample; current_sample; next_sample];
    end
    
    % Add current sample to the segmented data cell array
    segmented_dda{i} = current_sample;
end

clear i sample_size sample_start sample_end num_samples next_sample_start next_sample_end next_sample current_sample prev_sample_start prev_sample_end prev_sample

%  check if each cell contains 0. and number of 0 data 
num_samples = length(segmented_dda);
zero_counts = zeros(num_samples, 1);

for i = 1:num_samples
    zero_counts(i) = numel(segmented_dda{i}) - nnz(segmented_dda{i});
end

% disp(zero_counts);

clear i num_samples sagment_matrix

%  check numbers for each column of each cell 
% initialize zero_counts cell array
zero_colum_each = cell(size(segmented_dda));

% loop through each cell in segmented_dda
for i = 1:numel(segmented_dda)
    % get current cell
    current_cell = segmented_dda{i};
    
    % count number of zeros in each column
    num_zeros = sum(current_cell == 0);
    
    % store results in zero_counts cell array
    zero_colum_each{i} = num_zeros;
end

clear i num_zeros current_cell

%% delete contain 0 cells

delete_segmented_dda = segmented_dda;
num_samples = length(delete_segmented_dda);
zero_counts = zeros(num_samples, 1);

for i = 1:num_samples
    zero_counts(i) = numel(delete_segmented_dda{i}) - nnz(delete_segmented_dda{i});
    if zero_counts(i) > 0
        delete_segmented_dda{i} = []; % set the cell to empty
    end
end

delete_zero_data = delete_segmented_dda(~cellfun('isempty', delete_segmented_dda)); % remove empty cells

clear i num_samples

%  check if each cell contains 0

num_samples = length(delete_zero_data);
delete_nonzero_check = zeros(num_samples, 1);

for i = 1:num_samples
    delete_nonzero_check(i) = numel(delete_zero_data{i}) - nnz(delete_zero_data{i});
end

clear i num_samples

%%  if zero_counts > 10%, delete and check

nozero_segmented_dda = segmented_dda;

for i = 1:length(nozero_segmented_dda) % iterate through each double
    curr_double = nozero_segmented_dda{i};
    num_zeros = sum(curr_double == 0); % count number of zeros in each column
    for j = 1:size(curr_double, 2) % iterate through each column
        if num_zeros(j) > 180 % if number of zeros is greater than 180
            nozero_segmented_dda{i} = []; % delete the entire double
            break % move to the next double
        else
            % fill the zero values with the average of the same position column of the next double
            if any(curr_double(:, j) == 0)
                next_idx = i + 1;
                while next_idx <= length(nozero_segmented_dda) && all(nozero_segmented_dda{next_idx}(:, j) == 0)
                    next_idx = next_idx + 1; % find the next non-zero column
                end
                if next_idx > length(nozero_segmented_dda) % if all remaining doubles contain zeros in the current column
                    prev_idx = i - 1;
                    while prev_idx >= 1 && all(nozero_segmented_dda{prev_idx}(:, j) == 0)
                        prev_idx = prev_idx - 1; % find the previous non-zero column
                    end
                    if prev_idx >= 1 % if previous non-zero column is found
                        nozero_segmented_dda{i}(:, j) = mean(nozero_segmented_dda{prev_idx}(:, j)); % fill with average of previous non-zero column
                    end
                else % if non-zero column is found in the next doubles
                    nozero_segmented_dda{i}(curr_double(:, j) == 0, j) = mean(nozero_segmented_dda{next_idx}(nozero_segmented_dda{next_idx}(:, j) ~= 0, j)); % fill with average of next non-zero column
                end
            end
        end
    end
end


nozero_segmented_dat = nozero_segmented_dda;
idx = cellfun(@isempty, nozero_segmented_dat);
nozero_segmented_dat(idx) = [];

%  check numbers for each column of each cell 
% initialize zero_counts cell array
nozero_counts = cell(size(nozero_segmented_dda));

% loop through each cell in segmented_dda
for i = 1:numel(nozero_segmented_dda)
    % get current cell
    current_cell = nozero_segmented_dda{i};
    
    % count number of zeros in each column
    num_zeros = sum(current_cell == 0);
    
    % store results in zero_counts cell array
    nozero_counts{i} = num_zeros;
end

clear i idx j curr_double next_idx num_zeros i num_zeros current_cell segmented_dda zero_counts zero_colum_each nozero_segmented_dda nozero_counts

%% generation index and save !!!!! According to the sample size to set

% Create a cell array
nozero_segmented_dat_new = cell(104, 2);   % According to the sample size to set

% Copy the data from the first column of the original cell array to the first column of the new cell array
nozero_segmented_dat_new(:, 1) = nozero_segmented_dat;

% Set the second column of the new cell array to 1
nozero_segmented_dat_new(:, 2) = num2cell(ones(104, 1));  % According to the sample size to set

nozero_segmented_index = nozero_segmented_dat_new(:,2);

for i = 1:numel(nozero_segmented_index)
     if nozero_segmented_index{i} == 1
         nozero_segmented_index{i} = 4;
     end
 end

clear nozero_segmented_dat_new dat_matrix i

%% %%Sample generation 

segment_emg_all = [emg_WYH_normal1; emg_WYH_normal2; emg_WYH_normal3; emg_WYH_pronation1; emg_WYH_pronation2; 
    emg_WYH_supination1; emg_WYH_supination2; emg_WYH_supination3; emg_WYH_toein1; emg_WYH_toeout1; emg_WYH_toeout2;];
segment_index_all = [index_WYH_normal1; index_WYH_normal2; index_WYH_normal3; index_WYH_pronation1; index_WYH_pronation2; 
    index_WYH_supination1; index_WYH_supination2; index_WYH_supination3; index_WYH_toein1; index_WYH_toeout1; index_WYH_toeout2;];

%% 
csvwrite('/Users/chenzcha/Desktop/WorkSpace/GaitData_process/Feature/WYH_index.csv', segment_index_all);

%% filter

for i = 1:1763
     for j = 1:1800
         EMG{i}(j,:) = filter(Hd5_500, segment_emg_all{i}(j,:));
     end
end

clear i j

% 1*1763 - 1763*1
EMG_t = transpose(EMG); 

%% 1800*8 - 8*1800

num_cells = size(EMG_t, 1); % get the number of cells

for i = 1:num_cells % loop through each cell
    EMG_t{i} = EMG_t{i}.'; % transpose the data in the cell
end

 clear i num_cells

%% %% dualtree

for i = 1:1763
    for j = 1:8
        [AwavenorestA{i}(j,:),AwavenorestD{i}(j,:)] = dualtree(EMG_t{i}(j,:), 'Level', 3);
    end
end 
% for i =1:100
% [dtcwt_subject20AAA(:,i), dtcwt_subject20DDD(:,i)] = dualtree(aatest(:,i), 'Level', 3);
% end
for i = 1:1763
    AwavenorestD_level_1{i}(:,:) = AwavenorestD{i}{1}(:,:);
    AwavenorestD_level_2{i}(:,:) = AwavenorestD{i}{2}(:,:);
    AwavenorestD_level_3{i}(:,:) = AwavenorestD{i}{3}(:,:);
end

clear i AwavenorestD_level_3 AwavenorestD_level_2 AwavenorestD_level_1 AwavenorestD transpose_matrix

%% %% %% 18 & 15 & 12 & 8

for i = 1:1763
    %%% Mean absolute value 
    %AwaveMAV{i}(:,:) = mean(abs(AwavenorestA{i}));
    %%% Standard Deviation
    %AwaveSD{i}(:,:) = std(AwavenorestA{i});
    %%Skewness
    AwaveSew{i}(:,:) = skewness(AwavenorestA{i});
    %%Kurtosis
    AwaveKurto{i}(:,:) = kurtosis(AwavenorestA{i});

    for ii = 1:8
    %%Zero crossing
        AwaveZC{i}(:,ii) = jNewZeroCrossing(AwavenorestA{i}(ii,:));
    %%%Average energy
        %AwaveAE{i}(:,ii) = jAverageEnergy(AwavenorestA{i}(:,ii));
    %%Waveform length
        AwaveWaveform{i}(:,ii) = jWaveformLength(AwavenorestA{i}(ii,:));
    %%Maximum Fractal Length
        AwaveFractal{i}(:,ii) = jMaximumFractalLength(AwavenorestA{i}(ii,:));
    %%%% here for new methods
        %AwaveMAV{i}(:,ii) = jMeanAbsoluteValue(AwavenorestA{i}(:,ii));
        AwaveIENG{i}(:,ii) = jIntegratedEMG(AwavenorestA{i}(ii,:));
        %AwaveRMS{i}(:,ii) = jRootMeanSquare(AwavenorestA{i}(:,ii));
        %AwaveLOGD{i}(:,ii) = jLogDetector(AwavenorestA{i}(:,ii));
        %complex numbers
        %AwaveLOGCVAR{i}(:,ii) = jLogCoefficientOfVariation(AwavenorestA{i}(:,ii));
        AwaveLOGDABS{i}(:,ii) = jLogDifferenceAbsoluteMeanValue(AwavenorestA{i}(ii,:));
        %AwaveCVAR{i}(:,ii) = jCoefficientOfVariation(AwavenorestA{i}(:,ii));
        %AwaveDABS{i}(:,ii) = jDifferenceAbsoluteMeanValue(AwavenorestA{i}(:,ii));
        %AwaveDABSD{i}(:,ii) = jDifferenceAbsoluteStandardDeviationValue(AwavenorestA{i}(:,ii));
        %AwaveDVAR{i}(:,ii) = jDifferenceVarianceValue(AwavenorestA{i}(:,ii));
        AwaveEMAV{i}(:,ii) = jEnhancedMeanAbsoluteValue(AwavenorestA{i}(ii,:));
        %AwaveSSI{i}(:,ii) = jSimpleSquareIntegral(AwavenorestA{i}(:,ii));

    end
end

clear i ii 

%% %% 

for i = 1:1763
    for ii = 1:8
        Feature_vec{i}(ii,:) = [AwaveWaveform{i}(ii); AwaveFractal{i}(ii); 
            AwaveIENG{i}(ii); AwaveLOGDABS{i}(ii); AwaveEMAV{i}(ii)];
    end
end

clear i ii j 

% csvwrite('FeaVec5_F1.csv',Feature_vec{1});
% csvwrite('FeaVec5_F2.csv',Feature_vec{2});
% csvwrite('FeaVec5_F3.csv',Feature_vec{3});
% csvwrite('FeaVec5_F4.csv',Feature_vec{4});
% csvwrite('FeaVec5_F5.csv',Feature_vec{5});
% % csvwrite('FeaVec5_F6.csv',Feature_vec{6});
% % csvwrite('FeaVec5_F7.csv',Feature_vec{7});
% % csvwrite('FeaVec5_F8.csv',Feature_vec{8});

%% 
for i = 1:1763
    for ii = 1:8
        AwaveSew{i}(:,:) = skewness(AwavenorestA{i}(ii,:));
        AwaveKurto{i}(:,:) = kurtosis(AwavenorestA{i}(ii,:));
        AwaveWaveform{i}(:,:) = jWaveformLength(AwavenorestA{i}(ii,:));
        AwaveFractal{i}(:,:) = jMaximumFractalLength(AwavenorestA{i}(ii,:));
        AwaveIENG{i}(:,:) = jIntegratedEMG(AwavenorestA{i}(ii,:));
        AwaveLOGDABS{i}(:,:) = jLogDifferenceAbsoluteMeanValue(AwavenorestA{i}(ii,:));
        AwaveEMAV{i}(:,:) = jEnhancedMeanAbsoluteValue(AwavenorestA{i}(ii,:));
    end

end

clear i ii 

%% 































%% change all 0 data 
nozero_segmented_dda = segmented_dda;
% Iterate through each cell in the cell array
for i = 1:length(nozero_segmented_dda)
    % Iterate through each column in the data
    for j = 1:size(nozero_segmented_dda{i}, 2)
        % Check if the column contains any 0 values
        if any(nozero_segmented_dda{i}(:, j) == 0)
            % Find the next cell that has non-zero values in the same column
            k = i+1;
            while k <= length(nozero_segmented_dda) && all(nozero_segmented_dda{k}(:, j) == 0)
                k = k+1;
            end
            % If all subsequent cells have 0 values in the same column, fill with previous non-zero column average
            if k > length(nozero_segmented_dda)
                prev_i = i-1;
                while prev_i >= 1 && all(nozero_segmented_dda{prev_i}(:, j) == 0)
                    prev_i = prev_i-1;
                end
                nozero_segmented_dda{i}(nozero_segmented_dda{i}(:, j) == 0, j) = mean(nozero_segmented_dda{prev_i}(nozero_segmented_dda{prev_i}(:, j) ~= 0, j));
            % Otherwise, fill with average of next non-zero column
            else
                nozero_segmented_dda{i}(nozero_segmented_dda{i}(:, j) == 0, j) = mean(nozero_segmented_dda{k}(nozero_segmented_dda{k}(:, j) ~= 0, j));
            end
        end
    end
end
clear i num_zeros current_cell j k mean_val


%% zero valves imputation and check （Fill with the mean of the most recent sample without 0.）

% Find the indices of the zero values in segmented_dda{17}(:,1)
zero_indices = find(segmented_dda{40}(:,5) == 0);

% Impute the zero values with the mean of segmented_dda{15}(:,1)
mean_val = mean(segmented_dda{37}(:,5));
segmented_dda{40}(zero_indices,5) = mean_val;

% plot(segmented_dda{16}(:,1))

each_zero_counts = cell(size(segmented_dda));

% loop through each cell in segmented_dda
for i = 1:numel(segmented_dda)
    % get current cell
    current_cell = segmented_dda{i};
    
    % count number of zeros in each column
    num_zeros = sum(current_cell == 0);
    
    % store results in zero_counts cell array
    each_zero_counts{i} = num_zeros;
end

clear zero_indices i num_zeros
%% after zero valves imputation, check if each cell contains 0

num_samples = length(segmented_dda);
zero_counts = zeros(num_samples, 1);

for i = 1:num_samples
    zero_counts(i) = numel(segmented_dda{i}) - nnz(segmented_dda{i});
end

% disp(zero_counts);
clear i num_samples

%%   zero valves imputation  1
% (mean of segmented_dda{15}(:,1) to correctly impute the zero values of segmented_dda{17}(:,1))

test_signal1 = segmented_dda{17}(:,1);
test_signal2 = segmented_dda{15}(:,1);
% Find the indices of the zero values in test_signal1
zero_indices = find(test_signal1 == 0);
% Compute the mean or median of segmented_dda{15}(:,1)
impute_val = mean(test_signal2);
% Alternatively, you can use median instead of mean:
% impute_val = median(segmented_dda{15}(:,1));
% Impute the zero values with the computed value
test_signal1(zero_indices,1) = impute_val;
% plot(test_signal1)
% new
% figure
% plot(test_signal2)


%% 2: K-nearest neighbor imputation
% Load your data
signal = segmented_dda{17}(:,1)
% Find the indices of the zero values
zero_indices = find(signal == 0);
% Impute the zero values using k-nearest neighbor method
k = 100; % Choose the number of neighbors to consider
for i = 1:length(zero_indices)
    idx = zero_indices(i);
    % Find the indices of the k nearest non-zero values
    [~, nearest_indices] = mink(abs(zero_indices-idx), k);
    nearest_values = signal(zero_indices(nearest_indices));
    % Impute the zero value with the mean of the k nearest non-zero values
    signal(idx) = mean(nearest_values);
end

%%  K-nearest neighbor imputation followed by linear regression imputation
% Find the indices of the missing values
missing_indices = find(isnan(signal));

% Impute missing values using k-nearest neighbor method
k = 1000;
for i = 1:length(missing_indices)
    idx = missing_indices(i);
    % Find the indices of the k nearest non-missing values
    [~, nearest_indices] = mink(abs(missing_indices-idx), k);
    nearest_values = signal(missing_indices(nearest_indices));
    % Impute the missing value with the mean of the k nearest non-missing values
    signal(idx) = mean(nearest_values);
end

% Impute missing values using linear regression method
X = signal(:, 1:end-1); % Predictor variables
Y = signal(:, end); % Response variable
% Find the indices of the missing values
missing_indices = find(isnan(Y));
% Split data into complete and incomplete sets
complete_X = X(~isnan(Y), :);
complete_Y = Y(~isnan(Y));
incomplete_X = X(isnan(Y), :);
% Fit linear regression model
mdl = fitlm(complete_X, complete_Y);
% Predict missing values using the fitted model
predicted_Y = predict(mdl, incomplete_X);
% Impute missing values with predicted values
Y(isnan(Y)) = predicted_Y;
% Combine predictor and response variables
signal = [X Y];

%% Bayesian model averaging

% Generate some sample data with missing values
% data = [1 2 3; 4 NaN 6; 7 8 NaN; NaN 12 13];

% Set the number of imputations and models to use for Bayesian model averaging
num_imputations = 36;
num_models = 3;

% Preallocate an array to store the imputed datasets
imputed_data = cell(1, num_imputations);

% Perform Bayesian model averaging
for i = 1:num_imputations
    % Initialize a matrix to store the imputed values
    imputed_matrix = zeros(size(signal));
    
    % Iterate over each missing value in the dataset
    for j = 1:numel(signal)
        % If the value is missing, impute it using Bayesian model averaging
        if isnan(signal(j))
            % Generate multiple imputation models using different methods
            models = cell(1, num_models);
            models{1} = mean_imputation(signal);
            models{2} = k_nearest_neighbors(signal);
            models{3} = regression_imputation(signal);
            
            % Compute the posterior distribution over the missing value using Bayesian model averaging
            posterior = zeros(1, num_models);
            for k = 1:num_models
                posterior(k) = compute_posterior(models{k}, j);
            end
            posterior = posterior / sum(posterior);
            
            % Sample an imputed value from the posterior distribution
            imputed_matrix(j) = randsample(models{1}(j), 1, true, posterior);
        else
            % Copy over non-missing values from the original dataset
            imputed_matrix(j) = signal(j);
        end
    end
    
    % Add the imputed dataset to the array
    imputed_data{i} = imputed_matrix;
end

% Compute the average imputed dataset using Bayesian model averaging
average_imputed_data = mean(cat(3, imputed_data{:}), 3);


%% Sequential imputation:
% Load your data
% data = load('your_data.mat');
% Find the indices of the missing values
missing_indices = find(isnan(signal));
% Impute the missing values using simple imputation (e.g. mean imputation)
mean_val = nanmean(signal);
signal(missing_indices) = mean_val;
% Impute the remaining missing values using complex imputation (e.g. regression-based imputation)
complex_data = signal; % make a copy of the data to use for complex imputation
complex_missing_indices = find(isnan(complex_data));
% Use regression-based imputation to fill in missing values
for i = 1:length(complex_missing_indices)
    idx = complex_missing_indices(i);
    % Identify the predictor variables for the regression model
    predictor_indices = find(~isnan(complex_data(:,idx)));
    predictors = complex_data(predictor_indices,idx);
    % Identify the response variable for the regression model
    response = complex_data(predictor_indices,:);
    % Fit a regression model to predict the missing value
    mdl = fitlm(predictors,response);
    % Predict the missing value using the regression model
    predicted_val = predict(mdl,signal(predictor_indices,idx)');
    % Update the data matrix with the predicted value
    signal(idx) = predicted_val;
end


%% 
csvwrite('dat_matrix.csv',dat_matrix);

%% % define the filter
Fs = 1000;                                          % Sampling Frequency
Fpass = [5 500];                                    % Passband Frequencies
Fstop = [2.5 525];                                  % Stopband Frequencies
Rp =  1;                                            % Passband Ripple (dB)
Rs = 25;                                            % Stopband Attenuation (dB)
[b,a] = designfilt('bandpassiir', 'FilterOrder', 4, 'PassbandFrequency', Fpass, ...
    'StopbandFrequency', Fstop, 'PassbandRipple', Rp, 'StopbandAttenuation', Rs, ...
    'SampleRate', Fs);                              % Design IIR filter

% apply filter to each cell in data matrix
for i = 1:size(non_zero_data, 1)
    for j = 1:size(non_zero_data, 2)
        % apply filter to current cell
        current_data = non_zero_data{i,j};
        filtered_data = filter(b, a, current_data);
        % store filtered data in current cell
        non_zero_data{i,j} = filtered_data;
    end
end

clear a b d g i sos fs f_low f_high Fs Fn Wp Ws Rp Rs

%% 
for i = 8
    sEMG(:,i) = filter(Hd5_500, non_zero_data(:,i));
end

%% filter 5_500

Fs = 1000;                                          % Sampling Frequency
Fn = Fs/2;                                          % Nyquist Frequency
Wp = [10  495]/Fn;                                  % Norrmalised Passband Frequencies
Ws = [5   499]/Fn;                                  % Normalised Stopband Frequencies
Rp =  1;                                            % Passband Ripple (dB)
Rs = 25;                                            % Stopband Ripple (dB)
[n,Wn] = buttord(Wp, Ws, Rp, Rs);                   % Optimal Filter Order
[b,a] = butter(n, Wn);                              % Calculate Filter Coefficients
[sos,g] = tf2sos(b,a);                              % Convert To Second-Order Sections For Stability

test_data = table2array(test_data);                 % Convert to double matrix
filtered_data = filtfilt(sos, g, test_data);

figure(1)
freqz(sos, 4096, Fs)                                % Filter Bode Plot

% Plot the original and filtered data
figure;
subplot(2,1,1); plot(test_data); title('Original Data');
subplot(2,1,2); plot(filtered_data); title('Filtered Data');

%% segmentation

sample_size = 1600;
num_samples = floor(size(filtered_data,1) / sample_size);

samples = cell(num_samples,1);
for i = 1:num_samples
    sample_start = (i-1)*sample_size + 1;
    sample_end = i*sample_size;
    samples{i} = filtered_data(sample_start:sample_end,:);
end

% delete the remaining rows
delete_data1 = filtered_data(1:num_samples*sample_size,:);


%% moving window

% Set window size in seconds
window_size_sec = 0.2;

% Set sampling frequency in Hz
Fs = 1000;

% Calculate window size in samples
window_size = round(window_size_sec * Fs);

% Iterate through each cell in the data cell array
for i = 1:length(samples)
    % Get the current data sample
    sample = samples{i};
    
    % Initialize a matrix to store the processed data
    processed_data = zeros(size(sample));
    
    % Iterate through each column in the data sample
    for j = 1:size(sample, 2)
        % Apply the moving window to the current column of data
        processed_data(:, j) = movmean(sample(:, j), window_size);
    end
    
    % Store the processed data in the cell array
    MW_data{i} = processed_data;
end
%% 
DTCWT_data = MW_data; % replace 'my_data' with the name of your variable




%% dualtree

for i = 1:8
    [AwavenorestA{i}(:,:),AwavenorestD{i}(:,:)] = dualtree(MW_data{i}, 'Level', 3);
end 
% for i =1:100
% [dtcwt_subject20AAA(:,i), dtcwt_subject20DDD(:,i)] = dualtree(aatest(:,i), 'Level', 3);
% end
for i = 1:8
    AwavenorestD_level_1{i}(:,:) = AwavenorestD{i}{1}(:,:);
    AwavenorestD_level_2{i}(:,:) = AwavenorestD{i}{2}(:,:);
    AwavenorestD_level_3{i}(:,:) = AwavenorestD{i}{3}(:,:);
end

%% 
% Example code for Dual-Tree Complex Wavelet Transform

% Define wavelet and number of decomposition levels
wavelet = 'db4';
level = 5;

% Load data (1x113 cell array)
DTCW_data = MW_data;

% Perform transform on each cell
DTCWT_data = cell(size(DTCW_data));
for i = 1:numel(DTCW_data)
    % Perform 2D DTCWT
    [C, S] = dwt2(DTCW_data{i}, wavelet);
    for j = 1:level-1
        % Extract subbands from horizontal, vertical, and diagonal components
        H1 = detcoef2('h', C, S, j);
        V1 = detcoef2('v', C, S, j);
        D1 = detcoef2('d', C, S, j);
        H2 = detcoef2('h', H1, S(j+1));
        V2 = detcoef2('v', V1, S(j+1));
        D2 = detcoef2('d', D1, S(j+1));
        % Combine subbands and reconstruct
        A = idwt2(C, H2, V2, D2, wavelet);
        DTCWT_data{i} = cat(3, DTCWT_data{i}, A);
    end
end



%% 
window_size = 0.2 * 1000; % window size in samples (Fs is the sampling frequency)
MW_data = cell(1, length(samples)); % create an empty cell array to store the processed data
for i = 1:length(samples)
    MW_data{i} = movmean(samples{i}, window_size);
end

%% %% 
%  Delete any rows that don't fit into a sample of 1500 rows
n_samples = floor(size(filtered_data, 1) / 1500);
data = filtered_data(1:n_samples*1500, :);

%Reshape the data into a 3D array, with each "page" representing one sample
segmented_data = reshape(data', [8, 1500, n_samples]);


%% %% 
%%%% filter %%%%%
for i = 1:8
    data(:,i) = filter(Hd5_500, data(:,i));
end

%% 
startPoint = diff(restimulus);
startlabel = find(startPoint ~= 0);
startIndex = [0;startlabel];

%%% Segment the data and the index

for i = 1:length(startIndex)
    if i == length(startIndex)
        %%% for the last segmentation, we need read the data and label to
        %%% the end
        EMG_index{i}(:,:) = restimulus(startIndex(i)+1 : end);
         EMG_emg{i}(:,:) = EMG(startIndex(i)+1 : end, :);
    else
        %%% segment the label and data from the last one +1 to the next
        EMG_index{i}(:,:) = restimulus(startIndex(i)+1 : startIndex(i+1));
         EMG_emg{i}(:,:) = EMG(startIndex(i)+1 : startIndex(i+1), :);
    end

end

%%tranpose the cell
EMG_index = EMG_index';
EMG_emg = EMG_emg';

clear startIndex startlabel startPoint i EMG restimulus emg acc object reobject rerepetition repetition_object repetition stimulus

emg_200ms_sample = []; %%%sample rate is 2000, so 200ms is a 1*400 vector
emg_200ms_index = [];

for i = 1:length(EMG_emg)
    segment_emg = EMG_emg{i};
    segment_index = EMG_index{i};

    %%%check subject number is right?

    len = fix(length(segment_emg)/200);
    
    %%% 400 point (200ms), half overlapping
    for movw = 1:len-1
        %%segment emg signal
        temp = segment_emg(movw*200-199 : movw*200+200, :);
        emg_200ms_sample = [emg_200ms_sample ; temp];
        
        %%segment index signal
        temp_index = segment_index(movw*200-199 : movw*200+200, :);
        emg_200ms_index = [emg_200ms_index ; temp_index];
    end
end

% emg_200ms_sample(:,9) = [];
% em g_200ms_sample(:,9) = [];

clear EMG_emg EMG_index i len movw segment_emg segment_index temp temp_index

segment_emg_ = emg_200ms_sample;
segment_index_ = emg_200ms_index;

clear emg_200ms_sample emg_200ms_index subj time daytesting exercise glove inclin

%% %% Sample generation based on moving window
segment_emg_all = [segment_emg_sub1; segment_emg_sub2; segment_emg_sub3; segment_emg_sub4;
    segment_emg_sub5; segment_emg_sub8;];
segment_index_all = [segment_index_sub1; segment_index_sub2;segment_index_sub3;segment_index_sub4;
    segment_index_sub5;segment_index_sub8;];

 %% norest
%%%original is 64436
segment_emg_all_norest(:,:) = segment_emg_all(find(segment_index_all > 0),:);
segment_index_all_norest(:,:) = segment_index_all(find(segment_index_all > 0),:);
%%norest is 48117 for 400 label
bbb = reshape(segment_index_all_norest,[400,29255]);
bbbb = mean(bbb);
csvwrite('/Users/chenzcha/Desktop/DB3/Feature_data/NinaproDB3_sub123458_norest_400label.csv',bbbb);

% moving window 
for i = 1:12
    norestdat{i}(:,:) = reshape (segment_emg_all_norest (:,i), [400,29255]);
end
norestdat = norestdat';

%% dualtree
for i = 1:12
    [AwavenorestA{i}(:,:),AwavenorestD{i}(:,:)] = dualtree(norestdat{i}, 'Level', 3);
end 
% for i =1:100
% [dtcwt_subject20AAA(:,i), dtcwt_subject20DDD(:,i)] = dualtree(aatest(:,i), 'Level', 3);
% end
for i = 1:12
    AwavenorestD_level_1{i}(:,:) = AwavenorestD{i}{1}(:,:);
    AwavenorestD_level_2{i}(:,:) = AwavenorestD{i}{2}(:,:);
    AwavenorestD_level_3{i}(:,:) = AwavenorestD{i}{3}(:,:);
end

%% %% 18 & 15 & 12 & 8
for i = 1:12
    %%% Mean absolute value 
    %AwaveMAV{i}(:,:) = mean(abs(AwavenorestA{i}));
    %%% Standard Deviation
    %AwaveSD{i}(:,:) = std(AwavenorestA{i});
    %%Skewness
    AwaveSew{i}(:,:) = skewness(AwavenorestA{i});
    %%Kurtosis
    AwaveKurto{i}(:,:) = kurtosis(AwavenorestA{i});

    for ii = 1:29255
    %%Zero crossing
        AwaveZC{i}(:,ii) = jNewZeroCrossing(AwavenorestA{i}(:,ii));
    %%%Average energy
        %AwaveAE{i}(:,ii) = jAverageEnergy(AwavenorestA{i}(:,ii));
    %%Waveform length
        AwaveWaveform{i}(:,ii) = jWaveformLength(AwavenorestA{i}(:,ii));
    %%Maximum Fractal Length
        AwaveFractal{i}(:,ii) = jMaximumFractalLength(AwavenorestA{i}(:,ii));
    %%%% here for new methods
        %AwaveMAV{i}(:,ii) = jMeanAbsoluteValue(AwavenorestA{i}(:,ii));
        AwaveIENG{i}(:,ii) = jIntegratedEMG(AwavenorestA{i}(:,ii));
        %AwaveRMS{i}(:,ii) = jRootMeanSquare(AwavenorestA{i}(:,ii));
        %AwaveLOGD{i}(:,ii) = jLogDetector(AwavenorestA{i}(:,ii));
        %complex numbers
        %AwaveLOGCVAR{i}(:,ii) = jLogCoefficientOfVariation(AwavenorestA{i}(:,ii));
        AwaveLOGDABS{i}(:,ii) = jLogDifferenceAbsoluteMeanValue(AwavenorestA{i}(:,ii));
        %AwaveCVAR{i}(:,ii) = jCoefficientOfVariation(AwavenorestA{i}(:,ii));
        %AwaveDABS{i}(:,ii) = jDifferenceAbsoluteMeanValue(AwavenorestA{i}(:,ii));
        %AwaveDABSD{i}(:,ii) = jDifferenceAbsoluteStandardDeviationValue(AwavenorestA{i}(:,ii));
        %AwaveDVAR{i}(:,ii) = jDifferenceVarianceValue(AwavenorestA{i}(:,ii));
        AwaveEMAV{i}(:,ii) = jEnhancedMeanAbsoluteValue(AwavenorestA{i}(:,ii));
        %AwaveSSI{i}(:,ii) = jSimpleSquareIntegral(AwavenorestA{i}(:,ii));

    end
end
%% 
for i = 1:12
    Feature_vec{i} = [AwaveSew{i}; AwaveKurto{i}; AwaveZC{i}; 
        AwaveWaveform{i}; AwaveFractal{i}; AwaveIENG{i}; 
        AwaveLOGDABS{i}; AwaveEMAV{i}];
end

csvwrite('FeaVec8_DB3_norest_F1.csv',Feature_vec{1});
csvwrite('FeaVec8_DB3_norest_F2.csv',Feature_vec{2});
csvwrite('FeaVec8_DB3_norest_F3.csv',Feature_vec{3});
csvwrite('FeaVec8_DB3_norest_F4.csv',Feature_vec{4});
csvwrite('FeaVec8_DB3_norest_F5.csv',Feature_vec{5});
csvwrite('FeaVec8_DB3_norest_F6.csv',Feature_vec{6});
csvwrite('FeaVec8_DB3_norest_F7.csv',Feature_vec{7});
csvwrite('FeaVec8_DB3_norest_F8.csv',Feature_vec{8});
csvwrite('FeaVec8_DB3_norest_F9.csv',Feature_vec{9});
csvwrite('FeaVec8_DB3_norest_F10.csv',Feature_vec{10});
csvwrite('FeaVec8_DB3_norest_F11.csv',Feature_vec{11});
csvwrite('FeaVec8_DB3_norest_F12.csv',Feature_vec{12});

%% zero_index=find(test_data==0)

plot(segmented_dda{17}(:,1))
plot(segmented_dda{17}(:,2))
plot(segmented_dda{17}(:,3))
plot(segmented_dda{17}(:,4))
plot(segmented_dda{17}(:,5))
plot(segmented_dda{17}(:,6))
plot(segmented_dda{17}(:,7))
plot(segmented_dda{17}(:,8))


% test_signal = segmented_dda{17}(:,1);
% y = test_signal(non_zero_indices);
% mdl = fitlm(X, y);
% test_signal(zero_indices) = predict(mdl, zero_indices);
% plot(test_signal)

% 
% % Load your data
% signal = load('your_signal_data.mat');
% % Replace zeros with mean of non-zero values
% mean_val = mean(segmented_dda{17}(:,1)(segmented_dda{17}(:,1)~=0));
% test_signal(segmented_dda{17}(:,1)==0) = mean_val;