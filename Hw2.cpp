#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <tuple>

// --- Data Structures ---

/**
 * @brief Represents a coordinate location (row, col) in the grid.
 * The operators are overloaded to allow usage in STL containers like std::set.
 */
struct Location {
    int row, col;

    bool operator==(const Location& other) const {
        return row == other.row && col == other.col;
    }

    bool operator<(const Location& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

/**
 * @brief Represents a state in the problem space.
 * A state is defined by the agent's current location and the set of remaining dirty locations.
 * The operators are overloaded for use in the 'closed' set for graph search.
 */
struct State {
    Location agent_loc;
    std::set<Location> dirty_locs;

    bool operator==(const State& other) const {
        return agent_loc == other.agent_loc && dirty_locs == other.dirty_locs;
    }

    bool operator<(const State& other) const {
        if (!(agent_loc == other.agent_loc)) return agent_loc < other.agent_loc;
        return dirty_locs < other.dirty_locs;
    }
};

/**
 * @brief Represents a node in the search tree/graph.
 * It contains the state, a pointer to its parent for path reconstruction,
 * the action that led to this node, the total path cost (g(n)), and the depth.
 */
struct Node {
    State state;
    std::shared_ptr<Node> parent;
    std::string action;
    double path_cost;
    int depth;
};

/**
 * @brief Custom comparator for the priority queue (fringe) used in Uniform Cost Search.
 * It ensures that nodes with lower path_cost have higher priority.
 * It uses the agent's location (row, then column) as a tie-breaker, as specified.
 */
struct NodeComparator {
    bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) const {
        // Primary sort criterion: path_cost (ascending)
        if (std::abs(a->path_cost - b->path_cost) > 1e-9) { // Safe double comparison
            return a->path_cost > b->path_cost;
        }
        // Tie-breaker 1: agent's row number (ascending)
        if (a->state.agent_loc.row != b->state.agent_loc.row) {
            return a->state.agent_loc.row > b->state.agent_loc.row;
        }
        // Tie-breaker 2: agent's column number (ascending)
        return a->state.agent_loc.col > b->state.agent_loc.col;
    }
};

// Type alias for the fringe priority queue
using Fringe = std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodeComparator>;

/**
 * @class VacuumSolver
 * @brief Encapsulates the logic for solving the vacuum world problem.
 * It contains implementations for Uniform Cost Tree Search, Uniform Cost Graph Search,
 * and Iterative Deepening Tree Search.
 */
class VacuumSolver {
public:
    VacuumSolver(State initial_state, int rows, int cols)
        : initial_state_(initial_state), grid_rows_(rows), grid_cols_(cols) {
        action_costs_ = {
            {"Up", 0.8},
            {"Down", 0.7},
            {"Left", 1.0},
            {"Right", 0.9},
            {"Suck", 0.6}
        };
    }

    /**
     * @brief Runs all three search algorithms on the problem instance.
     */
    void solve() {
        std::cout << "Solving for initial state: ";
        print_state(initial_state_);
        std::cout << "\n\n";

        uniform_cost_tree_search();
        uniform_cost_graph_search();
        iterative_deepening_tree_search();
    }

private:
    State initial_state_;
    int grid_rows_;
    int grid_cols_;
    std::map<std::string, double> action_costs_;
    const int REPORT_FIRST_N_EXPANDED = 5;

    // --- Helper Functions ---

    /**
     * @brief Checks if a given state is the goal state (no dirt left).
     */
    bool is_goal(const State& state) const {
        return state.dirty_locs.empty();
    }

    /**
     * @brief Generates all possible successor nodes from a given parent node.
     */
    std::vector<std::shared_ptr<Node>> expand(const std::shared_ptr<Node>& parent_node) {
        std::vector<std::shared_ptr<Node>> successors;
        const auto& parent_state = parent_node->state;

        // Define movement actions: {name, d_row, d_col}
        std::vector<std::tuple<std::string, int, int>> moves = {
            {"Up", -1, 0}, {"Down", 1, 0}, {"Left", 0, -1}, {"Right", 0, 1}
        };

        // Generate successors for movement actions
        for (const auto& move : moves) {
            const auto& action_name = std::get<0>(move);
            Location new_loc = {
                parent_state.agent_loc.row + std::get<1>(move),
                parent_state.agent_loc.col + std::get<2>(move)
            };

            // Check if the move is within the grid boundaries
            if (new_loc.row >= 1 && new_loc.row <= grid_rows_ && new_loc.col >= 1 && new_loc.col <= grid_cols_) {
                auto successor = std::make_shared<Node>();
                successor->state.agent_loc = new_loc;
                successor->state.dirty_locs = parent_state.dirty_locs; // Dirt doesn't change on move
                successor->parent = parent_node;
                successor->action = action_name;
                successor->path_cost = parent_node->path_cost + action_costs_.at(action_name);
                successor->depth = parent_node->depth + 1;
                successors.push_back(successor);
            }
        }

        // Generate successor for "Suck" action
        auto suck_successor = std::make_shared<Node>();
        suck_successor->state = parent_state; // Location doesn't change
        // If agent is on a dirty square, clean it
        if (suck_successor->state.dirty_locs.count(parent_state.agent_loc)) {
            suck_successor->state.dirty_locs.erase(parent_state.agent_loc);
        }
        suck_successor->parent = parent_node;
        suck_successor->action = "Suck";
        suck_successor->path_cost = parent_node->path_cost + action_costs_.at("Suck");
        suck_successor->depth = parent_node->depth + 1;
        successors.push_back(suck_successor);

        return successors;
    }
    
    // --- Output and Reporting ---

    void print_state(const State& state) const {
        std::cout << "[V(" << state.agent_loc.row << "," << state.agent_loc.col << ")";
        for (const auto& dirt : state.dirty_locs) {
            std::cout << ", d(" << dirt.row << "," << dirt.col << ")";
        }
        std::cout << "]";
    }

    void print_first_n_expanded(const std::vector<State>& expanded_states) const {
        std::cout << "a. First " << REPORT_FIRST_N_EXPANDED << " expanded nodes' states:\n";
        for (int i = 0; i < expanded_states.size(); ++i) {
            std::cout << "   " << i + 1 << ". ";
            print_state(expanded_states[i]);
            std::cout << "\n";
        }
    }
    
    void print_solution(const std::shared_ptr<Node>& goal_node, long long nodes_expanded, long long nodes_generated, double duration_sec) const {
        std::vector<std::string> path;
        std::shared_ptr<Node> current = goal_node;
        while (current != nullptr && current->parent != nullptr) {
            path.push_back(current->action);
            current = current->parent;
        }
        std::reverse(path.begin(), path.end());

        std::cout << "b. Nodes expanded: " << nodes_expanded << "\n";
        std::cout << "   Nodes generated: " << nodes_generated << "\n";
        std::cout << "   CPU time: " << std::fixed << std::setprecision(4) << duration_sec << " seconds\n";
        
        std::cout << "c. Solution path:\n   ";
        for (size_t i = 0; i < path.size(); ++i) {
            std::cout << path[i] << (i == path.size() - 1 ? "" : ", ");
        }
        std::cout << "\n";
        std::cout << "   Number of moves: " << path.size() << "\n";
        std::cout << "   Solution cost: " << std::fixed << std::setprecision(1) << goal_node->path_cost << "\n";
    }
    
    // --- Search Algorithms ---

    void uniform_cost_tree_search() {
        std::cout << "--- Uniform Cost Tree Search ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        auto root = std::make_shared<Node>();
        root->state = initial_state_;
        root->parent = nullptr;
        root->path_cost = 0.0;
        root->depth = 0;

        Fringe fringe;
        fringe.push(root);

        long long nodes_generated = 1;
        long long nodes_expanded = 0;
        std::vector<State> first_expanded_states;

        while (!fringe.empty()) {
            std::shared_ptr<Node> current_node = fringe.top();
            fringe.pop();

            if (is_goal(current_node->state)) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end_time - start_time;
                print_first_n_expanded(first_expanded_states);
                print_solution(current_node, nodes_expanded, nodes_generated, duration.count());
                std::cout << "----------------------------------" << std::endl;
                return;
            }

            // Expand node
            nodes_expanded++;
            if (first_expanded_states.size() < REPORT_FIRST_N_EXPANDED) {
                first_expanded_states.push_back(current_node->state);
            }
            
            auto successors = expand(current_node);
            nodes_generated += successors.size();

            for (const auto& successor : successors) {
                fringe.push(successor);
            }
        }
        std::cout << "No solution found." << std::endl;
    }

    void uniform_cost_graph_search() {
        std::cout << "--- Uniform Cost Graph Search ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        auto root = std::make_shared<Node>();
        root->state = initial_state_;
        root->parent = nullptr;
        root->path_cost = 0.0;
        root->depth = 0;

        Fringe fringe;
        fringe.push(root);
        std::set<State> closed_set;

        long long nodes_generated = 1;
        long long nodes_expanded = 0;
        std::vector<State> first_expanded_states;

        while (!fringe.empty()) {
            std::shared_ptr<Node> current_node = fringe.top();
            fringe.pop();
            
            if (is_goal(current_node->state)) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end_time - start_time;
                print_first_n_expanded(first_expanded_states);
                print_solution(current_node, nodes_expanded, nodes_generated, duration.count());
                std::cout << "----------------------------------" << std::endl;
                return;
            }

            // Only expand if the state has not been visited before
            if (closed_set.find(current_node->state) == closed_set.end()) {
                closed_set.insert(current_node->state);
                
                nodes_expanded++;
                if (first_expanded_states.size() < REPORT_FIRST_N_EXPANDED) {
                    first_expanded_states.push_back(current_node->state);
                }

                auto successors = expand(current_node);
                nodes_generated += successors.size();

                for (const auto& successor : successors) {
                    fringe.push(successor);
                }
            }
        }
        std::cout << "No solution found." << std::endl;
    }

    // Recursive helper for Iterative Deepening
    std::shared_ptr<Node> depth_limited_search(const std::shared_ptr<Node>& node, int limit, long long& expanded_count, long long& generated_count) {
        if (is_goal(node->state)) {
            return node;
        }

        if (limit == 0) {
            return nullptr; // Cutoff
        }

        expanded_count++;
        auto successors = expand(node);
        generated_count += successors.size();

        for (const auto& successor : successors) {
            auto result = depth_limited_search(successor, limit - 1, expanded_count, generated_count);
            if (result != nullptr) {
                return result;
            }
        }

        return nullptr;
    }

    void iterative_deepening_tree_search() {
        std::cout << "--- Iterative Deepening Tree Search ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        long long total_nodes_generated = 0;
        long long total_nodes_expanded = 0;

        for (int depth_limit = 0; ; ++depth_limit) {
            long long expanded_in_iter = 0;
            long long generated_in_iter = 1; // root

            auto root = std::make_shared<Node>();
            root->state = initial_state_;
            root->parent = nullptr;
            root->path_cost = 0.0;
            root->depth = 0;
            
            // This algorithm does not track first 5 expanded nodes as it re-expands them each iteration.
            // For simplicity and adherence to the spirit of IDTS, we report only the final metrics.
            std::shared_ptr<Node> result = depth_limited_search(root, depth_limit, expanded_in_iter, generated_in_iter);

            total_nodes_expanded += expanded_in_iter;
            total_nodes_generated += generated_in_iter;

            if (result != nullptr) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end_time - start_time;
                std::cout << "a. First 5 expanded nodes: Not tracked for IDTS due to repeated expansions.\n";
                print_solution(result, total_nodes_expanded, total_nodes_generated, duration.count());
                std::cout << "----------------------------------" << std::endl;
                return;
            }
            if (depth_limit > 50) { // Safety break for very deep solutions
                std::cout << "Search stopped after exceeding depth limit of 50." << std::endl;
                break;
            }
        }
        std::cout << "No solution found." << std::endl;
    }
};


int main() {
    // --- Instance #1 ---
    std::cout << "==========================\n";
    std::cout << "       INSTANCE 1\n";
    std::cout << "==========================\n";
    State initial_state_1;
    initial_state_1.agent_loc = {2, 2};
    initial_state_1.dirty_locs = {{1, 2}, {2, 4}, {3, 5}};
    VacuumSolver solver1(initial_state_1, 4, 5);
    solver1.solve();
    std::cout << "\n";

    // --- Instance #2 ---
    std::cout << "==========================\n";
    std::cout << "       INSTANCE 2\n";
    std::cout << "==========================\n";
    State initial_state_2;
    initial_state_2.agent_loc = {3, 2};
    initial_state_2.dirty_locs = {{1, 2}, {2, 1}, {2, 4}, {3, 3}};
    VacuumSolver solver2(initial_state_2, 4, 5);
    solver2.solve();

    return 0;
}