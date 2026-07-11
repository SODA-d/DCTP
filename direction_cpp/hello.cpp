#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cmath>
#include <utility>
#include <algorithm>

namespace py = pybind11;

std::pair<int, int> find_nearest_passable(const std::vector<std::vector<int>>& grid, int start_x, int start_y) {
    const std::vector<std::pair<int, int>> directions = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };

    int rows = grid.size();
    if (rows == 0) return {start_x, start_y};
    int cols = grid[0].size();

    // 安全越界判断
    if (start_x < 0 || start_x >= rows || start_y < 0 || start_y >= cols) {
        printf("%d,,%d\n",start_x,rows);
    }

    if (grid[start_x][start_y] == 0) {
        return {start_x, start_y};
    }

    std::queue<std::pair<int, int>> q;
    std::set<std::pair<int, int>> visited;

    q.push({start_x, start_y});
    visited.insert({start_x, start_y});

    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();

        for (const auto& [dx, dy] : directions) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && visited.count({nx, ny}) == 0) {
                if (grid[nx][ny] == 0) {
                    return {nx, ny};
                }
                visited.insert({nx, ny});
                q.push({nx, ny});
            }
        }
    }

    // 没找到就返回原地
    return {start_x, start_y};
}

void mark_direction(std::pair<int, int> agents_xy,
                    std::pair<float, float> target_xy,
                    py::array_t<float> &obs1,
                    py::list &median_point_xy,
                    int height, 
                    int width)
{
    int agent_x = agents_xy.first;
    int agent_y = agents_xy.second;

    int target_x = static_cast<int>(target_xy.first);
    int target_y = static_cast<int>(target_xy.second);

    int dx_abs = target_x - agent_x;
    int dy_abs = target_y - agent_y;
    int grid_x = 5 + dx_abs;
    int grid_y = 5 + dy_abs;

    int dx = grid_x - 5;
    int dy = grid_y - 5;

    if (dx == 0 && dy == 0)
        return;

    int edge_x = 0, edge_y = 0;

    if (dx == 0)
    {
        edge_y = (dy > 0) ? 10 : 0;
        edge_x = 5;
    }
    else if (dy == 0)
    {
        edge_x = (dx > 0) ? 10 : 0;
        edge_y = 5;
    }
    else
    {
        float tx = (dx > 0) ? (10.0f - 5.0f) / dx : (0.0f - 5.0f) / dx;
        float ty = (dy > 0) ? (10.0f - 5.0f) / dy : (0.0f - 5.0f) / dy;
        float t = (tx > 0 && ty > 0) ? std::min(tx, ty) : std::max(tx, ty);
        edge_x = static_cast<int>(std::round(5 + dx * t));
        edge_y = static_cast<int>(std::round(5 + dy * t));
        edge_x = std::max(0, std::min(10, edge_x));
        edge_y = std::max(0, std::min(10, edge_y));
    }

    auto buf = obs1.mutable_unchecked<2>();  // 2D array
    buf(edge_x, edge_y) = 2.0f;

    edge_x += agent_x;
    edge_y += agent_y;

    // 🔥 直接修改 Python 传入的 list
    median_point_xy[0] = std::max(5, std::min(edge_x, height));
    median_point_xy[1] = std::max(5, std::min(edge_y, width));
}

 

void process_observations(py::list observations,
                          const std::vector<std::pair<int, int>>& agents_xy,
                          const std::vector<std::vector<int>>& obstacles
){
    int height = obstacles.size()+4;
    int width = obstacles[0].size()+4;
    for(size_t agent_idx = 0; agent_idx < py::len(observations); ++agent_idx){
        py::dict obs_1 = observations[agent_idx];
        std::vector<std::pair<int, int>> communicate_target;

        for (size_t i = 0; i < py::len(observations); ++i){
            py::dict obs_2 = observations[i];
            py::list vis_targets = obs_2["visibled_targets"].cast<py::list>();
            py::list target = vis_targets[agent_idx].cast<py::list>();
            int target_x = target[0].cast<int>();
            int target_y = target[1].cast<int>();
            int agent_x = agents_xy[agent_idx].first;
            int agent_y = agents_xy[agent_idx].second;
            if (!(target_x == 0 && target_y == 0) && !(target_x == agent_x && target_y == agent_y)) {
                // 防止重复：遍历 communicate_target 并检查是否已包含相同的 target
                bool is_duplicate = false;
                for (const auto& t : communicate_target) {
                    if (t.first == target_x && t.second == target_y) {
                        is_duplicate = true;
                        break;
                    }
                }
                // 如果没有重复，添加到 communicate_target
                if (!is_duplicate) {
                    communicate_target.push_back(std::make_pair(target_x, target_y));
                }
            }
        }
        
        py::list median_point_xy;
        median_point_xy.append(0);
        median_point_xy.append(0);

        for (auto communicate_target_j : communicate_target){
            float tx = static_cast<float>(communicate_target_j.first);
            float ty = static_cast<float>(communicate_target_j.second);
            py::array_t<float> agents_array = obs_1["agents"].cast<py::array_t<float>>();
            mark_direction(agents_xy[agent_idx], {tx, ty},agents_array, median_point_xy,height,width);
            printf("median_point_xy: %d, %d\n", median_point_xy[0].cast<int>(), median_point_xy[1].cast<int>());
            int a_x = median_point_xy[0].cast<int>();
            int b_y = median_point_xy[1].cast<int>();

            auto [px, py_] = find_nearest_passable(obstacles, a_x - 5, b_y - 5);

            px += 5;
            py_ += 5;

            py::list xy = obs_1["xy"].cast<py::list>();  // 将 "xy" 转换为 py::list
            int result_x = px - agents_xy[agent_idx].first + xy[0].cast<int>();  // 获取第一个元素并转换为 int
            int result_y = py_ - agents_xy[agent_idx].second + xy[1].cast<int>();  // 获取第二个元素并转换为 int

            obs_1["edge_target_xy"] = py::make_tuple(result_x, result_y);
        }
        if (!obs_1.contains("edge_target_xy") || obs_1["edge_target_xy"].cast<std::pair<int, int>>() == std::make_pair(0, 0))
        {
            obs_1["edge_target_xy"] = obs_1["xy"];
        }

    }
}


PYBIND11_MODULE(Helloworld, m) {
    m.def("find_nearest_passable", &find_nearest_passable, "Find nearest free cell");
    m.def("mark_direction", &mark_direction, "Mark direction and return nearest passable global point");
    m.def("process_observations", &process_observations, "process_observations");
}
