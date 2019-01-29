function create_data(out_folder,s,num_samples)

    %create data directory if it doesnot exists
    if ~isfolder(out_folder)
        mkdir(out_folder);
    end

    %Transformation Matrix A
    
    A = randn(100,200);
    norm_A = vecnorm(A);
    A = A./norm_A;
    assert(all(int8(vecnorm(A))==1),'Matrix A is not Normalized');
    dlmwrite(strcat(out_folder,'matrix.txt'),A,'delimiter','\t');
    
    obs_id = strcat(out_folder,'observations.txt');
    inp_id = strcat(out_folder,'sparse_inputs.txt');
    for i = 1: num_samples
    
        % Sparse Vector X
        binary_mask = zeros(size(A,2),1);
        rand_indexes = randperm(size(A,2),s);
        binary_mask(rand_indexes) = 1;
        X =  binary_mask .* randn(size(A,2),1);


        %Observation Vector Y
        Y = A*X;
        
        if i == 1
           dlmwrite(obs_id,Y.','delimiter','\t');
           dlmwrite(inp_id,X.','delimiter','\t');
        else
            dlmwrite(obs_id,Y.','-append','delimiter','\t');
            dlmwrite(inp_id,X.','-append','delimiter','\t');
        end
       
    end
    
    file_id = fopen(strcat(out_folder,'parameters.txt'),'w');
    out = 'Number of Sparse Digits in the input : %d \nMatrix Size : %d x %d \nNumber of Samples : %d \n'
    fprintf(file_id,out,s,size(A,1),size(A,2),num_samples);
    fclose(file_id);

end