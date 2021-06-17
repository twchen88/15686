function [] = test_learn_hopfield_net(  )

load('test_data.mat');

w_students = learn_hopfield_net(test1);

if(isequal(w_students, w1))
    disp 'Test passed'
else
    disp 'Test failed.'
end

end

