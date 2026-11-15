% Initialize the Network
N = 50; % Number of nodes
X_max = 100; % Max X coordinate
Y_max = 100; % Max Y coordinate
positions = [rand(N,1)*X_max, rand(N,1)*Y_max]; % Node positions
initial_energy = 10; % Initial energy in Joules
energy = ones(N,1) * initial_energy; % Energy array

% Create the Distance Matrix
distance_matrix = zeros(N,N);
for i = 1:N
    for j = i+1:N
        distance_matrix(i,j) = sqrt((positions(i,1)-positions(j,1))^2 + (positions(i,2)-positions(j,2))^2);
        distance_matrix(j,i) = distance_matrix(i,j); % Symmetric matrix
    end
end

% Compute the Minimum Spanning Tree (MST)
G = graph(distance_matrix);
mst = minspantree(G);

% Simulate Energy Consumption
energy_per_unit_distance = 0.01; % Energy consumed per unit distance
for edge = table2array(mst.Edges)
    s = edge(1);
    t = edge(2);
    energy_used = distance_matrix(s,t) * energy_per_unit_distance;
    energy(s) = energy(s) - energy_used; % Deduct energy from sender
    energy(t) = energy(t) - energy_used; % Deduct energy from receiver
end

% Visualization
figure;
plot(G, 'XData', positions(:,1), 'YData', positions(:,2), 'NodeLabel',{});
hold on;
highlight(plot(mst), 'EdgeColor','r','LineWidth',1.5); % Highlight MST edges
title('Wireless Sensor Network with MST');
xlabel('X Position');
ylabel('Y Position');
hold off;

% Save Results
save('network_data.mat', 'positions', 'energy', 'mst');
savefig('network_figure.fig');
