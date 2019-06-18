function Turing_Texons_1_0(filename)
%采用由Python得到的由层级tree产生刺激图片

try
    %定义相关参数=================================================
    %电脑设置要求，1024*768,85hz刷新率
    %定义相关参数----------------------------
    
    global screenNumber w wRect frame a b
    screenNumber = max(Screen('Screens'));
    Screen('Resolution', screenNumber, [1024], [768], [60]);
    cm2pixel = 28;%1cm换成呈现像素的单位大小
    pic_num = 48;
    
    %-------------------------------------------------
    %Open screen---------------------------------------
    [w, wRect]=Screen('OpenWindow',screenNumber, [0,0,0],[],32,2);
    frame_duration = Screen('GetFlipInterval',w);
    Screen(w,'BlendFunction',GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    frame = round(1/frame_duration);
    [a,b]=WindowCenter(w);
    Screen('BlendFunction', w, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    % Bump priority for speed
    priorityLevel=MaxPriority(w);
    Priority(priorityLevel);
    
    %========================================================
    %========================================
    %按键设置
    KbName('UnifyKeyNames');
    J = KbName('J');
    F = KbName('F');
    confirm =  KbName('Space');
    escapekey = KbName('escape');
    RestrictKeysForKbCheck([27,32,70,74]); %KbCheck时仅检测esc、J、F、Space四个按键
    %==================读入图片============================
    start_img = imread('Pic\start.jpg');
    instruction_img = imread('Pic\instruction.png');
    rest_img = imread('Pic\pic_wait.jpg');
    over_img = imread('Pic\over.jpg');
    mask_img = imread('Pic\mask.png');
    start = Screen('MakeTexture',w,start_img);
    instruction = Screen('MakeTexture',w,instruction_img);
    rest = Screen('MakeTexture',w,rest_img);
    over = Screen('MakeTexture',w,over_img);
    mask = Screen('MakeTexture',w,mask_img);
    startRect = Screen('Rect', start);
%     instructionRect = Screen('Rect',instruction);
    instructionRect = [0,0,1024,768];
    restRect = Screen('Rect', rest);
    overRect = Screen('Rect', over);
    maskRect = Screen('Rect', mask);
    HideCursor;
    Response = ones(1, pic_num)*-1;
    Response_time = zeros(1, pic_num);
    Type = zeros(1, pic_num);
    penwidth = 12;
    
    %==============================================
    %===================实验部分开始===============================
    HideCursor;
    Screen('DrawTexture', w, instruction, [],[a-instructionRect(3)/2 b-instructionRect(4)/2 a+instructionRect(3)/2 b+instructionRect(4)/2]);
    Screen('Flip',w);
    KbWait;
    keyisdown = 1;
    while(keyisdown) % first wait until all keys are released
        [keyisdown,secs,keycode] = KbCheck;
        WaitSecs(0.001); % delay to prevent CPU hogging
    end
    Screen('DrawTexture', w, start, [],[a-startRect(3)/2 b-startRect(4)/2 a+startRect(3)/2 b+startRect(4)/2]);
    Screen('Flip',w);
    KbWait;
    %     time1=GetSecs;
    %=========================================
    Screen('TextSize',w,40);
    pic_name = [0, 3, 5, 6, 8, 14, 17, 20, 22, 23, 30, 31, 32, 35, 37, 38, 39, 41, 43, 44, 45, 47, 48, 49, 51, 53, 54, 57, 59, 60, 63, 69, 112, 123, 130, 136, 147, 160, 171, 176, 200, 207, 208, 213, 220, 223, 235, 239];
    
    
    random_list = randperm(pic_num);
    
    for trial = 1:pic_num
        mmp = randperm(2);
        Type(trial) = mmp(1);
        random_list(trial)
        if Type(trial) == 1
            img = imread(['answers/demo',num2str(pic_name(random_list(trial))),'_', filename, '.png']); 
        end
        
        if Type(trial) == 2
            img = imread(['answers/demo',num2str(pic_name(random_list(trial))),'_', filename, 'humanAnswer.png']);
        end
        
        img_texture = Screen('MakeTexture',w,img);
        imgRect = Screen('Rect', img_texture) * 1;
        Screen('DrawTexture',w,img_texture,[],[a-imgRect(3)/2 b-imgRect(4)/2 a+imgRect(3)/2 b+imgRect(4)/2]);        
        Screen('Flip',w);
        
        start_time = GetSecs;
        while GetSecs - start_time<90
            [keyisdown,secs,keycode] = KbCheck;
            if keycode(J)                       
                Response(trial) = 1;
                Response_time(trial) = GetSecs - start_time;
                break
            end
            if keycode(F)                      
                Response(trial) = 0;
                Response_time(trial) = GetSecs - start_time;
                break
            end
            if keycode(escapekey)
                Response(trial) = 9;
                Response_time(trial) = GetSecs - start_time;
                break
            end
        end
        time1=GetSecs;
        
        Screen('Flip',w);
        if keycode(escapekey)
            break
        end
        keyisdown = 1;
        while(keyisdown) % first wait until all keys are released
            [keyisdown,secs,keycode] = KbCheck;
            WaitSecs(0.001); % delay to prevent CPU hogging
        end
        WaitSecs(0.6+rand*0.4);   %调节trial间间隔时间
%         WaitSecs(0.05);
        %----------------------------------------------------------------
        %====================================休息七次
%         if trial==40||trial==80||trial==120||trial==160 
%             Screen('DrawTexture', w, rest, [], [a-restRect(3)/2 b-restRect(4)/2 a+restRect(3)/2 b+restRect(4)/2]);
%             Screen('Flip',w);
%             keyisdown = 1;
%             while(keyisdown) % first wait until all keys are released
%                 [keyisdown,secs,keycode] = KbCheck;
%                 WaitSecs(0.001); % delay to prevent CPU hogging
%             end
%             KbWait;
%         end;
        Screen('Flip',w);
    end % end for trial
    %实验结束=================================================
    Screen('Flip',w);
    WaitSecs(1.5);
    Priority(0);
    Screen('Close',w);
    ShowCursor;
    fid=fopen(['Results\','Turing_Texons_1_0_',filename,'.txt'],'w');
    fprintf(fid,'%s\t%s\t%s\t%s\t%s\t%s\n','Sub','Trial','img','type','RT','Reaction');
    for n=1:trial
        fprintf(fid,'%s\t%f\t%f\t%f\t%6.3f\t%f\n',filename,n,pic_name(random_list(n)),Type(n),Response_time(n)*1000,Response(n));
    end
    fclose(fid);
    clear all
    %-----------------------------------------------
    % 错误返回语句
catch
    Screen('Closeall')
    rethrow(lasterror)
    clear all
end
%%----------------------------------------------
