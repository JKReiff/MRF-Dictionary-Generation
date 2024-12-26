% Define the number of patients

NumbPatients = 1000; %input how many patients do you want to generate Dictionaries for 

% Define the path to save the data
path_save = '/Users/janreifferscheidt/Documents/Data_Generator';

% Call the TrainDataGenerator function with the specified inputs
TrainDataGenerator(NumbPatients, path_save);


function TrainDataGenerator(NumbPatients, path_save)
    %% variable INPUT
 
    Dictlist = 'T1T2_lowfield_invivo'; % Here is defined how many combinations of t1 and t2 and so on should be generated look in the siumaltion_function folder under loadDictionarylist_v4
    End = NumbPatients;  % Assuming input1 is End value
    folder_path = path_save;  % Assuming input2 is folder_path
    
  
    %%
    warning('off');
    addpath(genpath("./"))
    csv_path = "csvdata/mrf_15HB_TFE25_5_30_Cardiac_2echo.csv"; % sequence csv
    RR_matrix = importdata('RR_Matrix_Full.txt'); %import the RR-Intervall Matrix which is generated in 0_RR_intevall_sim
    
    %% %%%%%%%%%% to update start when not all files where created (checks folder if 1000
    files = dir(fullfile(folder_path, 'RR_*.h5'));  
    highest_number = 0;  
    
    for idx = 1:length(files)
        file_name = files(idx).name;
        % Extract the number from the file name
        number_str = regexp(file_name, 'RR_(\d+).h5', 'tokens');
        if ~isempty(number_str)
            number = str2double(number_str{1}{1});
            % Update the highest number
            if number > highest_number
                highest_number = number;
            end
        end
    end
    
    start = highest_number + 1;
    %%  
    
    for i = start : End   
        disp(['File: ', num2str(i)]);
        row_data = RR_matrix(i, :);
        extra_args.Acq.RR = row_data; % Set the 'row' variable in extra_args
        extra_args.Acq.balanced = 0; % this is to define if the sequence is GRE or bssfp
        [~, ~, GenDict] = simulate_newdict(csv_path, Dictlist, extra_args);
        dic = GenDict.dictOn;
        combi = GenDict.Combinations;
        INDEX_unphysCombis = find(combi(:, 2) > combi(:, 1));
        zeroRowsIndices = find(all(dic == 0, 2));
        real_part = real(dic);
        imag_part = imag(dic);
        dic = [real_part, imag_part];
    
    
        if i ==1  % in first file extracting the Combinations of T1 and T2 
            dict_path = fullfile(folder_path, 'T1_T2_Combinations.csv');
            combi_new = combi(:,1:2) ;
            combi_new(INDEX_unphysCombis, :) = [];
            writematrix(combi_new, dict_path);
            fprintf("Saved: T1_T2_Combinations");
    
            % Check for unphysical T Combis and delets them in Fingerprints
            if isequal(INDEX_unphysCombis, zeroRowsIndices)   
                path = fullfile(folder_path, sprintf("RR_%d.h5", i));
                dic(zeroRowsIndices, :) = [];
                h5create(path, "/dic", size(dic));
                h5write(path, "/dic", dic);
                %writematrix(dic, filename1);
                %save(filename1, 'dic'); load('data.mat');
                fprintf("Saved: RR_%d\n",i);
            else
                disp(['Stop: NOT equal for RR_' num2str(i)]);
                return;
            end
    
         
        else          
            % Check for unphysical T Combis and delets them in Fingerprints
            if isequal(INDEX_unphysCombis, zeroRowsIndices)            
                dic(zeroRowsIndices, :) = [];
                path = fullfile(folder_path, sprintf("RR_%d.h5", i));
                h5create(path, "/dic", size(dic));
                h5write(path, "/dic", dic);
                %writematrix(dic, path);
                % save(filename, 'dic');
                fprintf("Saved: RR_%d\n",i);
            else
                disp(['Stop: NOT equal for RR_' num2str(i)]);
                return;
            end
        
        end
    end
    
    
    
    
    %% Generate Input Data 
    
    filename = fullfile(folder_path, "/T1_T2_Combinations.csv");
    T1T2 = load(filename);
    length_combi = length(T1T2);
    
    for i = 1: End
        
        % Assuming RR_matrix is already defined
        
        % Extract the first row from RR_matrix
        firstRow = RR_matrix(i, :);
        
        % Repeat the first row to create a matrix with 1460 rows
        newMatrix = repmat(firstRow, length_combi, 1);
        params = [T1T2, newMatrix];
        filename = sprintf("%s/Params_RR_%d.h5", folder_path, i);
        h5create(filename, "/dic", size(params));
        h5write(filename, "/dic", params);
        %writematrix(params, filename);
                 
        disp(['File: ', num2str(i)]);
    
    end
end
 