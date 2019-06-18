function paired_comparision_1_1(filename)
%采用由Python得到的由层级tree产生刺激图片，进行变化觉察的实验。
Screen('Preference', 'SkipSyncTests', 1)
try
    %定义相关参数=================================================
    %电脑设置要求，1024*768,85hz刷新率
    %定义相关参数----------------------------
    
    global screenNumber w wRect frame a b
    screenNumber = max(Screen('Screens'));
    Screen('Resolution', screenNumber, [1024], [768], [60]);
    cm2pixel = 28;%1cm换成呈现像素的单位大小
    pic_num = 20;
    
    %-------------------------------------------------
    %Open screen---------------------------------------
    [w, wRect]=Screen('OpenWindow',screenNumber, [128,128,128],[],32,2);
    frame_duration = Screen('GetFlipInterval',w);
    Screen(w,'BlendFunction',GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    frame = round(1/frame_duration);
    [a,b]=WindowCenter(w);
    Screen('BlendFunction', w, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    % Bump priority for speed
    priorityLevel=MaxPriority(w);
    Priority(priorityLevel);
    type = cell(pic_num * (pic_num - 1) / 2, 2);
    %========================================================
    %========================================
    %按键设置
    KbName('UnifyKeyNames');
    J = KbName('J');
    F = KbName('F');
    confirm =  KbName('Space');
    escapekey = KbName('escape');
    %RestrictKeysForKbCheck([27,32,70,74]); %KbCheck时仅检测esc、J、F、Space四个按键
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
    Response = ones(1, pic_num * (pic_num - 1) / 2)*-1;
    Response_time = zeros(1, pic_num * (pic_num - 1) / 2);
    Pos_ran = zeros(1, pic_num * (pic_num - 1) / 2);
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
    pic_name = [9,14,15,25,26,30,31,36,38,39,45,46,47,48,50,51,54,55,57,59];
    temp = 1;
    for i = 1:19
        for j = i + 1:20
            type{temp, 1} = i;
            type{temp, 2} = j;
            temp = temp + 1;
        end
    end
    
    random_list = randperm(pic_num * (pic_num - 1) / 2)
    
    for trial = 1:pic_num * (pic_num - 1) / 2
        pos_random = randperm(2);
        Pos_ran(trial) = pos_random(1);
        tuple_num = random_list(trial);
        
        img1 = imread(['Stimular/demo',num2str(pic_name(type{tuple_num, 1})), '.png']); %左侧
        img2 = imread(['Stimular/demo',num2str(pic_name(type{tuple_num, 2})), '.png']); %右侧
        img1_texture = Screen('MakeTexture',w,img1);
        img2_texture = Screen('MakeTexture',w,img2);
        img1Rect = Screen('Rect', img1_texture) * 0.4;
        img2Rect = Screen('Rect', img1_texture) * 0.4;

        if pos_random(1) == 1
            Screen('DrawTexture',w,img1_texture,[],[a-img1Rect(3)-0.1*a b-img1Rect(4)/2 a-0.1*a b+img1Rect(4)/2]);
            Screen('DrawTexture',w,img2_texture,[],[a+0.1*a b-img2Rect(4)/2 a+img2Rect(3)+0.1*a b+img2Rect(4)/2]);
        end
        if pos_random(1) == 2
            Screen('DrawTexture',w,img2_texture,[],[a-img1Rect(3)-0.1*a b-img1Rect(4)/2 a-0.1*a b+img1Rect(4)/2]);
            Screen('DrawTexture',w,img1_texture,[],[a+0.1*a b-img2Rect(4)/2 a+img2Rect(3)+0.1*a b+img2Rect(4)/2]);
        end
        
        Screen('Flip',w);
        start_time = GetSecs;
        while GetSecs - start_time<30
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
%         WaitSecs(0.6+rand*0.4);   %调节trial间间隔时间
        WaitSecs(0.05);
        %----------------------------------------------------------------
        %====================================休息
        if trial==40||trial==80||trial==120||trial==160 
            Screen('DrawTexture', w, rest, [], [a-restRect(3)/2 b-restRect(4)/2 a+restRect(3)/2 b+restRect(4)/2]);
            Screen('Flip',w);
            keyisdown = 1;
            while(keyisdown) % first wait until all keys are released
                [keyisdown,secs,keycode] = KbCheck;
                WaitSecs(0.001); % delay to prevent CPU hogging
            end
            KbWait;
        end;
        Screen('Flip',w);
    end % end for trial
    %实验结束=================================================
    Screen('Flip',w);
    WaitSecs(1.5);
    Priority(0);
    Screen('Close',w);
    ShowCursor;
    fid=fopen(['Results\','paired_comparision_1_1_',filename,'.txt'],'w');
    fprintf(fid,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n','Sub','Trial','img1','img2','pos_ran','RT','Reaction');
    for n=1:trial
        fprintf(fid,'%s\t%f\t%f\t%f\t%f\t%6.3f\t%f\n',filename,n,type{random_list(n),1},type{random_list(n),2},Pos_ran(n),Response_time(n)*1000,Response(n));
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
