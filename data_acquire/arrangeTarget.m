function [tar_value, new_tar_pool] = arrangeTarget(tar_pool)
    tar_value = datasample(tar_pool,1);
    new_tar_pool = tar_pool;
    new_tar_pool(new_tar_pool==tar_value) = [];
end