path="C:\Users\Anoud\Desktop\rkrkr";
mD =fullfile(path,'701_StillsRaw_full');
xx = imageDatastore(mD);
I2 = readimage(xx,80);
I2 = histeq(I2);

I3 = readimage(xx,85);
I3 = histeq(I3);

subplot(2,2,1);
imshow(I2)
subplot(2,2,3);
imshow(I3)
classes = [
    "background"
    "hand"];
labelIDs1 =  camvidPixelLabelIDs(); 

lD = fullfile(path,'LabeledApproved_full');
pxds1 = pixelLabelDatastore(lD,classes,labelIDs1);


C2 = readimage(pxds1,80);
cmap1=camvidColorMap;

B2 = labeloverlay(I2,C2,'ColorMap',cmap1);
imshow(B2)

C3 = readimage(pxds1,85);
B3 = labeloverlay(I3,C3,'ColorMap',cmap1);
imshow(B3)

I1 = readimage(xx,80);
I4 = readimage(xx,85);

load trainedSystem
C1 = semanticseg(I1, trainedSystem);
B1 = labeloverlay(I1,C1,'Colormap',cmap1,'Transparency',0);
subplot(2,2,2);
imshow(B1)
C4 = semanticseg(I4, trainedSystem);
B4 = labeloverlay(I4,C4,'Colormap',cmap1,'Transparency',0);
subplot(2,2,4);
imshow(B4)

trainedSystem.HandleVisibility





%metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
%print("1");

%metrics.DataSetMetrics
%print("2");
%metrics.ClassMetrics
%print("3");

%net = resnet18();
%figure
%plot(net)
%title('Architecture of ResNet-50')
%set(gca, 'YLim',[150 170]);

function labelIDs = camvidPixelLabelIDs()
%Return the label IDs corresponding to each class.

labelIDs = {[ 000 000 000; ]
             [255 255 255; ]};
  
end

function cmap = camvidColorMap()
% Define the colormap used by CamVid dataset.

cmap = [000 000 000 
        255 255 255 ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end

