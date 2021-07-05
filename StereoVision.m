hIdtc = vision.ImageDataTypeConverter;
hCsc = vision.ColorSpaceConverter('Conversion','RGB to intensity');
leftI3chan = step(hIdtc,imread('LeftUndistRect2.jpg'));
leftI = step(hCsc,leftI3chan);
rightI3chan = step(hIdtc,imread('RightUndistRect2.jpg'));
rightI = step(hCsc,rightI3chan);

figure(1), clf;
imshow(rightI3chan), title('Right image');

figure(2), clf;
imshowpair(rightI,leftI,'ColorChannels','red-cyan'), axis image;
title('Color composite (right=red, left=cyan)');
Dbasic = zeros(size(leftI), 'single');
disparityRange = 15;
% Selects (2*halfBlockSize+1)-by-(2*halfBlockSize+1) block.
halfBlockSize = 3;
blockSize = 2*halfBlockSize+1;
% Allocate space for all template matcher System objects.
tmats = cell(blockSize);

% Initialize progress bar
hWaitBar = waitbar(0, 'Performing basic block matching...');
nRowsLeft = size(leftI, 1);

% Scan over all rows.
for m=1:nRowsLeft
    % Set min/max row bounds for image block.
    minr = max(1,m-halfBlockSize);
    maxr = min(nRowsLeft,m+halfBlockSize);
    % Scan over all columns.
    for n=1:size(leftI,2)
        minc = max(1,n-halfBlockSize);
        maxc = min(size(leftI,2),n+halfBlockSize);
        % Compute disparity bounds.
        mind = max( -disparityRange, 1-minc );
        maxd = min( disparityRange, size(leftI,2)-maxc );

        % Construct template and region of interest.
        template = rightI(minr:maxr,minc:maxc);
        templateCenter = floor((size(template)+1)/2);
        roi = [minc+templateCenter(2)+mind-1 ...
               minr+templateCenter(1)-1 ...
               maxd-mind+1 1];
        % Lookup proper TemplateMatcher object; create if empty.
        if isempty(tmats{size(template,1),size(template,2)})
            tmats{size(template,1),size(template,2)} = ...
                vision.TemplateMatcher('ROIInputPort',true);
        end
        thisTemplateMatcher = tmats{size(template,1),size(template,2)};

        % Run TemplateMatcher object.
        loc = step(thisTemplateMatcher, leftI, template, roi);
        Dbasic(m,n) = loc(1) - roi(1) + mind;
    end
    waitbar(m/nRowsLeft,hWaitBar);
end

close(hWaitBar);
figure(3), clf;
imshow(Dbasic,[]), axis image, colormap('jet'), colorbar;
caxis([0 disparityRange]);
title('Depth map from basic block matching');
DbasicSubpixel= zeros(size(leftI), 'single');
tmats = cell(2*halfBlockSize+1);
hWaitBar=waitbar(0,'Performing sub-pixel estimation...');
for m=1:nRowsLeft
    % Set min/max row bounds for image block.
    minr = max(1,m-halfBlockSize);
    maxr = min(nRowsLeft,m+halfBlockSize);
    % Scan over all columns.
    for n=1:size(leftI,2)
        minc = max(1,n-halfBlockSize);
        maxc = min(size(leftI,2),n+halfBlockSize);
        % Compute disparity bounds.
        mind = max( -disparityRange, 1-minc );
        maxd = min( disparityRange, size(leftI,2)-maxc );

        % Construct template and region of interest.
        template = rightI(minr:maxr,minc:maxc);
        templateCenter = floor((size(template)+1)/2);
        roi = [minc+templateCenter(2)+mind-1 ...
               minr+templateCenter(1)-1 ...
               maxd-mind+1 1];
        % Lookup proper TemplateMatcher object; create if empty.
        if isempty(tmats{size(template,1),size(template,2)})
            tmats{size(template,1),size(template,2)} = ...
                vision.TemplateMatcher('ROIInputPort',true,...
                'BestMatchNeighborhoodOutputPort',true);
        end
        thisTemplateMatcher = tmats{size(template,1),size(template,2)};

        % Run TemplateMatcher object.
        [loc,a2] = step(thisTemplateMatcher, leftI, template, roi);
        ix = single(loc(1) - roi(1) + mind);

        % Subpixel refinement of index.
        DbasicSubpixel(m,n) = ix - 0.5 * (a2(2,3) - a2(2,1)) ...
            / (a2(2,1) - 2*a2(2,2) + a2(2,3));
    end
    waitbar(m/nRowsLeft,hWaitBar);
end

close(hWaitBar);
figure(1), clf;
imshow(DbasicSubpixel,[]), axis image, colormap('jet'), colorbar;
caxis([0 disparityRange]);
title('Basic block matching with sub-pixel accuracy');
Ddynamic = zeros(size(leftI), 'single');
finf = 1e3; % False infinity
disparityCost = finf*ones(size(leftI,2), 2*disparityRange + 1, 'single');
disparityPenalty = 0.5; % Penalty for disparity disagreement between pixels
hWaitBar = waitbar(0,'Using dynamic programming for smoothing...');
% Scan over all rows.
for m=1:nRowsLeft
    disparityCost(:) = finf;
    % Set min/max row bounds for image block.
    minr = max(1,m-halfBlockSize);
    maxr = min(nRowsLeft,m+halfBlockSize);
    % Scan over all columns.
    for n=1:size(leftI,2)
        minc = max(1,n-halfBlockSize);
        maxc = min(size(leftI,2),n+halfBlockSize);
        % Compute disparity bounds.
        mind = max( -disparityRange, 1-minc );
        maxd = min( disparityRange, size(leftI,2)-maxc );
        % Compute and save all matching costs.
        for d=mind:maxd
            disparityCost(n, d + disparityRange + 1) = ...
                sum(sum(abs(leftI(minr:maxr,(minc:maxc)+d) ...
                - rightI(minr:maxr,minc:maxc))));
        end
    end

    % Process scan line disparity costs with dynamic programming.
    optimalIndices = zeros(size(disparityCost), 'single');
    cp = disparityCost(end,:);
    for j=size(disparityCost,1)-1:-1:1
        % False infinity for this level
        cfinf = (size(disparityCost,1) - j + 1)*finf;
        % Construct matrix for finding optimal move for each column
        % individually.
        [v,ix] = min([cfinf cfinf cp(1:end-4)+3*disparityPenalty;
                      cfinf cp(1:end-3)+2*disparityPenalty;
                      cp(1:end-2)+disparityPenalty;
                      cp(2:end-1);
                      cp(3:end)+disparityPenalty;
                      cp(4:end)+2*disparityPenalty cfinf;
                      cp(5:end)+3*disparityPenalty cfinf cfinf],[],1);
        cp = [cfinf disparityCost(j,2:end-1)+v cfinf];
        % Record optimal routes.
        optimalIndices(j,2:end-1) = (2:size(disparityCost,2)-1) + (ix - 4);
    end
    % Recover optimal route.
    [~,ix] = min(cp);
    Ddynamic(m,1) = ix;
    for k=1:size(Ddynamic,2)-1
        Ddynamic(m,k+1) = optimalIndices(k, ...
            max(1, min(size(optimalIndices,2), round(Ddynamic(m,k)) ) ) );
    end
    waitbar(m/nRowsLeft, hWaitBar);
end
close(hWaitBar);
Ddynamic = Ddynamic - disparityRange - 1;
figure(3), clf;
imshow(Ddynamic,[]), axis image, colormap('jet'), colorbar;
caxis([0 disparityRange]);
title('Block matching with dynamic programming');
% Construct a four-level pyramid
pyramids = cell(1,4);
pyramids{1}.L = leftI;
pyramids{1}.R = rightI;
for i=2:length(pyramids)
    hPyr = vision.Pyramid('PyramidLevel',1);
    pyramids{i}.L = single(step(hPyr,pyramids{i-1}.L));
    pyramids{i}.R = single(step(hPyr,pyramids{i-1}.R));
end
% Declare original search radius as +/-4 disparities for every pixel.
smallRange = single(3);
disparityMin = repmat(-smallRange, size(pyramids{end}.L));
disparityMax = repmat( smallRange, size(pyramids{end}.L));
% Do telescoping search over pyramid levels.
for i=length(pyramids):-1:1
    Dpyramid = vipstereo_blockmatch(pyramids{i}.L,pyramids{i}.R, ...
        disparityMin,disparityMax,...
        false,true,3,...
        true,... % Waitbar
        ['Performing level-',num2str(length(pyramids)-i+1),...
        ' pyramid block matching...']); %Waitbar title

    if i > 1
        % Scale disparity values for next level.
        hGsca = vision.GeometricScaler(...
            'InterpolationMethod','Nearest neighbor',...
            'SizeMethod','Number of output rows and columns',...
            'Size',size(pyramids{i-1}.L));
        Dpyramid = 2*step(hGsca, Dpyramid);
        % Maintain search radius of +/-smallRange.
        disparityMin = Dpyramid - smallRange;
        disparityMax = Dpyramid + smallRange;
    end
end

figure(3), clf;
imshow(Dpyramid,[]), colormap('jet'), colorbar, axis image;
caxis([0 disparityRange]);
title('Four-level pyramid block matching');
DpyramidDynamic = vipstereo_blockmatch_combined(leftI,rightI, ...
    'NumPyramids',3, 'DisparityRange',4, 'DynamicProgramming',true, ...
    'Waitbar', true, ...
    'WaitbarTitle', 'Performing combined pyramid and dynamic programming');
figure(3), clf;
imshow(DpyramidDynamic,[]), axis('image'), colorbar, colormap jet;
caxis([0 disparityRange]);
title('4-level pyramid with dynamic programming');

DdynamicSubpixel = vipstereo_blockmatch_combined(leftI,rightI, ...
    'NumPyramids',3, 'DisparityRange',4, 'DynamicProgramming',true, ...
    'Subpixel', true, ...
    'Waitbar', true, ...
    'WaitbarTitle', ['Performing combined pyramid and dynamic ',...
    'programming with sub-pixel estimation']);

figure(4), clf;
imshow(DdynamicSubpixel,[]), axis image, colormap('jet'), colorbar;
caxis([0 disparityRange]);
title('Pyramid with dynamic programming and sub-pixel accuracy');
% Camera matrix
K = [409.4433         0    0
            0  416.0865    0
     204.1225  146.4133    1];

% Create a sub-sampled grid for backprojection.
dec = 2;
[X,Y] = meshgrid(1:dec:size(leftI,2),1:dec:size(leftI,1));
P = [X(:), Y(:), ones(numel(X), 1, 'single')] / K;
Disp = max(0,DdynamicSubpixel(1:dec:size(leftI,1),1:dec:size(leftI,2)));
hMedF = vision.MedianFilter('NeighborhoodSize',[5 5]);
Disp = step(hMedF,Disp); % Median filter to smooth out noise.
% Derive conversion from disparity to depth with tie points:
knownDs = [15   9   2]'; % Disparity values in pixels
knownZs = [4  4.5 6.8]';
% World z values in meters based on scene measurements.
ab = [1./knownDs ones(size(knownDs), 'single')] \ knownZs; % least squares
% Convert disparity to z (distance from camera)
ZZ = ab(1)./Disp(:) + ab(2);
% Threshold to [0,8] meters.
ZZdisp = min(8,max(0, ZZ ));
Pd = bsxfun(@times,P,ZZ);
% Remove near points
bad = Pd(:, 3)>8 | Pd(:, 3)<3;
Pd = Pd(~bad, :);
% Collect quantized colors for point display
Colors = rightI3chan(1:dec:size(rightI,1),1:dec:size(rightI,2),:);
Colors = reshape(Colors,[size(Colors,1)*size(Colors,2) size(Colors,3)]);
Colors = Colors(~bad,:);
cfac = 20;
C8 = round(cfac*Colors);
[U,I,J] = unique(C8,'rows');
C8 = C8/cfac;

figure(2), clf, hold on, axis equal;
for i=1:size(U,1)
    plot3(-Pd(J==i,1),-Pd(J==i,3),-Pd(J==i,2),'.','Color',C8(I(i),:));
end
view(161,14), grid on;
xlabel('x (meters)'), ylabel('z (meters)'), zlabel('y (meters)');
