% % Run PALM analyses with different combinations of data, contrast and
% design matrices This script performs PALM analyses across different
% hemispheres, measures, design variables and area variables.

% Set your working path here.
basePath = '...';

% Construct paths relative to the script directory
baseDataPath = fullfile(basePath, 'data');
baseOutPath = fullfile(basePath, 'results');

% Create the results directory if it doesn't exist
if ~exist(baseOutPath, 'dir')
    mkdir(baseOutPath);
    fprintf('Created results directory: %s\n', baseOutPath);
end


% Validate paths exist
if ~exist(baseOutPath, 'dir') || ~exist(baseDataPath, 'dir')
    error('Base directories do not exist. Please check paths.');
end

% Define analysis parameters
hemispheres  = {'lh', 'rh'};
measures     = {'thickness', 'volume', 'area'};
designVars   = {'',};
overwrite = false; % Set to true to force reprocessing

% Run PALM analyses
for hemisphere = hemispheres
    for measure = measures
        for designVar = designVars
            % Construct file paths
            fsFile = fullfile(baseDataPath, ...
                sprintf('%s_%s.csv', hemisphere{1},  measure{1}));

            designFile = fullfile(baseDataPath, ...
                sprintf('design_%s%s.csv', measure{1}, designVar{1}));

            contrastFile = fullfile(baseDataPath, ...
                sprintf('contrast_%s.csv', measure{1}));

            % Construct output paths
            outFile = sprintf('%s_%s%s', ...
                hemisphere{1}, measure{1}, designVar{1});
            outPath = fullfile(baseOutPath, outFile, outFile);
            outDir = fullfile(baseOutPath, outFile);
            
            % Skip if output directory exists and overwrite is false
            if exist(outDir, 'dir') && ~overwrite
                warning('Output directory already exists, skipping: %s', outDir);
                continue;
            end

            % Validate input files exist
            if ~exist(fsFile, 'file')
                warning('Input file missing: %s', fsFile);
                continue;
            end

            if ~exist(designFile, 'file')
                warning('Design file missing: %s', designFile);
                continue;
            end

            if ~exist(contrastFile, 'file')
                warning('Contrast file missing: %s', contrastFile);
                continue;
            end

            % Uncomment to execute PALM
            % fprintf('Running PALM analysis for: %s\n', outFile);
            cmd = sprintf('palm -i %s -d %s -t %s -corrmod -corrcon -fdr -twotail -demean -saveglm -seed 0 -o %s', fsFile, designFile, contrastFile, outPath);
            eval(cmd);
        end
    end
end