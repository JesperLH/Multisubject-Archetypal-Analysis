function noise = generateSyntheticNoise(sx,sy,noise_var,stepsize,show_plot)


if nargin < 5
    show_plot = false;
    if nargin < 4
        stepsize = 8;
        if nargin < 3
            noise_var = [1,2,4].^2;
        else
            error('Map sizes must be specified')
        end
    end
    
end

%Center
cx = sx/2+.5;
cy = sy/2+.5;
rx = sx/2;
ry = sy/2; %TODO make contours in y axis also 

if show_plot
  %  fig1 = figure('Position',[100,100,2000,1200]);
    fig2 = figure('Position',[100,100,2000,1200]);
end

%%
num_noise = length(noise_var);
noise=zeros(sx,sy,num_noise); % The resulting noise levels
for j = 1:num_noise
    X = zeros(sx,sy);
    %Draw noise contours
    noise_contours = rx:-stepsize:1; %From out to in
    noise_var_con = linspace(0.01*noise_var(j),noise_var(j),length(noise_contours));
    for i = 1:length(noise_contours),
        [rr, cc] = meshgrid(1:sx);
        C = sqrt((rr-cx).^2+(cc-cy).^2)<=noise_contours(i);
        X(C) = noise_var_con(i);
    end
    
    %Assign to be returned
    noise(:,:,j) = X;
    
    %Illustration
    if show_plot
%         colormap gray;
%         figure(fig1)
%         subplot(3,3,j); imagesc(X);  axis([0,sx,0,sy]);   colorbar; title(sprintf('Max variance %4.2f',noise_var(j))) ;
%         subplot(3,3,j+3); imagesc(sqrt(X));  axis([0,sx,0,sy]);   colorbar;  title(sprintf('Max standard deviation %4.2f',sqrt(noise_var(j)))) ;
%         subplot(3,3,j+6); imagesc(randn(sx,sy).*sqrt(X));  axis([0,sx,0,sy]);   colorbar; title(sprintf('Normal distributed noise , N(0,%i)',noise_var(j))) ; caxis([0,max(noise_var)])

        figure(fig2)
        colormap gray;
        X(X==0) = nan;
        subplot(3,num_noise,j); h=pcolor(X);  axis([0,sx,0,sy]);   colorbar; title(sprintf('Max variance %4.2f',noise_var(j))) ;set(h,'edgecolor','none');
        subplot(3,num_noise,j+num_noise); h=pcolor(sqrt(X));  axis([0,sx,0,sy]);   colorbar;  title(sprintf('Max standard deviation %4.2f',sqrt(noise_var(j)))) ;set(h,'edgecolor','none');
        subplot(3,num_noise,j+num_noise*2); h=pcolor(randn(sx,sy).*sqrt(X));  axis([0,sx,0,sy]);   colorbar; title(sprintf('Normal distributed noise , N(0,%i)',noise_var(j))) ;set(h,'edgecolor','none'); caxis([0,max(noise_var)])
    end
    
end
end