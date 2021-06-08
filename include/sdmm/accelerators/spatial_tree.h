#pragma once

#include <atomic>
#include <vector>

#include <sdmm/core/utils.h>
#include <sdmm/linalg/aabb.h>

namespace sdmm::accelerators {

template <typename Scalar, int Size, typename Value>
struct STreeNode {
    using AABB = sdmm::linalg::AABB<Scalar, Size>;
    using Point = typename AABB::Point;

    STreeNode()
        : is_leaf(true), axis(0), children{0, 0}, value(nullptr), depth(-1) {}

    Value* find(
        const Point& point,
        const std::vector<STreeNode>& nodes,
        AABB& found_aabb) const {
        if (is_leaf) {
            found_aabb = aabb;
            // spdlog::info(
            //     "aabb.min={}, aabb.max={}, data_aabb.min={},
            //     data_aabb.max={}", aabb.min, aabb.max, data_aabb.min,
            //     data_aabb.max
            // );
            return value.get();
        }

        int found_child = -1;
        for (int child_i = 0; child_i < 2; ++child_i) {
            const int child_idx = children[child_i];
            if (child_i == 0) {
                if (nodes[child_idx].aabb.min.coeff(axis) < point.coeff(axis)) {
                    found_child = child_idx;
                    break;
                }
            } else {
                found_child = child_idx;
            }
        }
        assert(found_child != -1);
        if (auto found = nodes[found_child].find(point, nodes, found_aabb);
            found != nullptr) {
            return found;
        }
        return nullptr;
    }

    bool is_leaf = true;
    int axis = 0;
    uint32_t idx = 0;
    std::array<uint32_t, 2> children;
    std::unique_ptr<Value> value;
    AABB aabb;
    int depth;
};

// template<typename Scalar, int Size, typename Value>
// void to_json(json& j, const STreeNode<Scalar, Size, Value>& node) {
//     j = json{{"is_leaf", node.is_leaf}, {"axis", node.axis}};
// }

// template<typename Scalar, int Size, typename Value>
// STreeNode<Scalar, Size, Value> from_json(const json& j) {
//     STreeNode<Scalar, Size, Value> node;
//     j.at("is_leaf").get_to(node.is_leaf);
//     j.at("axis").get_to(node.axis);
//     return std::move(node);
// }

template <typename Scalar, int Size, typename Value>
class STree {
   public:
    using Node = STreeNode<Scalar, Size, Value>;
    using AABB = typename Node::AABB;
    using Point = typename AABB::Point;

    STree() = default;
    STree(const AABB& aabb, std::unique_ptr<Value> value) : m_aabb(aabb) {
        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Point diagonal = m_aabb.diagonal();
        m_aabb.max = m_aabb.min + enoki::full<Point>(enoki::hmax(diagonal));

        m_nodes.resize(64000);
        m_nodes[0].aabb = m_aabb;
        m_nodes[0].value = std::move(value);
        m_nodes[0].depth = 0;
        m_nodes_end.store(1);
    }

    Value* find(const Point& point) const {
        AABB aabb;
        return m_nodes[0].find(point, m_nodes, aabb);
    }

    Value* find(const Point& point, AABB& aabb) const {
        return m_nodes[0].find(point, m_nodes, aabb);
    }

    auto begin() {
        return m_nodes.begin();
    }

    auto end() {
        return *m_nodes[m_nodes_end.load()];
    }

    auto size() {
        return m_nodes_end.load();
    }

    const auto& data() {
        return m_nodes;
    }

    const auto& aabb() {
        return m_aabb;
    }

    std::pair<int, Scalar> get_split_location(int node_i) {
        auto& stats = m_nodes[node_i].value->stats;
        auto mean_point = stats.mean_point();
        auto mean_sqr_point = stats.mean_sqr_point();
        float stats_size = (float) stats.size;
        auto var_point = (mean_sqr_point -
                          enoki::sqr(mean_point / stats_size) * stats_size) /
            (float)(stats_size - 1);
        mean_point /= stats_size;
        mean_sqr_point /= stats_size;

        size_t max_var_i = 0;
        for (size_t var_i = 0; var_i < 3; ++var_i) {
            assert(std::isfinite(var_point.coeff(var_i)));
            if (var_point.coeff(var_i) > var_point.coeff(max_var_i)) {
                max_var_i = var_i;
            }
        }

        Point aabb_min = m_nodes[node_i].aabb.min;
        Point aabb_diagonal = m_nodes[node_i].aabb.diagonal();
        Scalar location =
            (mean_point.coeff(max_var_i) - aabb_min.coeff(max_var_i)) /
            aabb_diagonal.coeff(max_var_i);
        return {max_var_i, location};
    }

    Node create_child(int node_i, int child_i, Scalar splitLocation) {
        Node child;

        // Set correct parameters for child node
        child.is_leaf = true;
        int axis = m_nodes[node_i].axis;
        child.axis = (axis + 1) % Size;
        child.aabb = m_nodes[node_i].aabb;
        // child.data_aabb = m_nodes[node_i].data_aabb;
        if (child_i == 0) {
            child.aabb.min.coeff(axis) +=
                splitLocation * child.aabb.diagonal().coeff(axis);
            // child.data_aabb.min.coeff(axis) = child.aabb.min.coeff(axis);
        } else {
            child.aabb.max.coeff(axis) -=
                (1.f - splitLocation) * child.aabb.diagonal().coeff(axis);
            // child.data_aabb.max.coeff(axis) = child.aabb.max.coeff(axis);
        }

        auto childValue =
            std::make_unique<Value>(m_nodes[node_i].value->data.capacity);

        childValue->sdmm = m_nodes[node_i].value->sdmm;
        childValue->conditioner = m_nodes[node_i].value->conditioner;
        childValue->em = m_nodes[node_i].value->em;
        childValue->initialized = m_nodes[node_i].value->initialized;
        for (size_t sample_i = 0; sample_i < m_nodes[node_i].value->data.size;
             ++sample_i) {
            Point point(
                enoki::slice(
                    m_nodes[node_i].value->data.point.coeff(0), sample_i),
                enoki::slice(
                    m_nodes[node_i].value->data.point.coeff(1), sample_i),
                enoki::slice(
                    m_nodes[node_i].value->data.point.coeff(2), sample_i));
            if (child.aabb.contains(point)) {
                childValue->data.push_back(
                    enoki::slice(m_nodes[node_i].value->data, sample_i));
                childValue->stats.push_back(point);
            }
        }
        child.value = std::move(childValue);

        return child;
    }

    void split_to_depth(int max_depth) {
        split_to_depth_recurse(0, 0, max_depth);
    }

    void split_to_depth_recurse(uint32_t node_i, int depth, int max_depth) {
        int max_axis = (depth == 0) ? Size : 3;
        int next_depth =
            (m_nodes[node_i].axis == max_axis - 1) ? (depth + 1) : depth;
        if (!m_nodes[node_i].is_leaf) {
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_to_depth_recurse(
                    m_nodes[node_i].children[child_i], next_depth, max_depth);
            }
            return;
        }

        if (depth >= max_depth) {
            return;
        }

        m_nodes[node_i].is_leaf = false;
        for (int child_i = 0; child_i < 2; ++child_i) {
            // Insert new child into vector
            uint16_t child_idx = m_nodes_end.fetch_add(1);
            m_nodes[child_idx] = create_child(node_i, child_i, 0.5);
            m_nodes[child_idx].idx = child_idx;
            m_nodes[child_idx].depth = m_nodes[node_i].depth + 1;
            m_nodes[node_i].children[child_i] = child_idx;
        }

        m_nodes[node_i].value->data = enoki::zero<decltype(Value::data)>(0);
        m_nodes[node_i].value->training_data =
            enoki::zero<decltype(Value::training_data)>(0);
        m_nodes[node_i].value->em = enoki::zero<decltype(Value::em)>(0);
        m_nodes[node_i].value = nullptr;

        for (int child_i = 0; child_i < 2; ++child_i) {
            int child_idx = m_nodes[node_i].children[child_i];
            split_to_depth_recurse(child_idx, next_depth, max_depth);
        }
    }

    void split_leaf_recurse(uint32_t node_i, size_t split_threshold) {
        if (!m_nodes[node_i].is_leaf ||
            m_nodes[node_i].value->stats.size <= split_threshold) {
            return;
        }

        m_nodes[node_i].is_leaf = false;
        std::pair<int, Scalar> split = get_split_location(node_i);
        m_nodes[node_i].axis = split.first;
        for (int child_i = 0; child_i < 2; ++child_i) {
            // Insert new child into vector
            uint16_t child_idx = m_nodes_end.fetch_add(1);
            m_nodes[child_idx] = create_child(node_i, child_i, split.second);
            m_nodes[child_idx].idx = child_idx;
            m_nodes[child_idx].depth = m_nodes[node_i].depth + 1;
            m_nodes[node_i].children[child_i] = child_idx;
        }
        m_nodes[node_i].value->stats.clear();
        m_nodes[node_i].value->data.clear();
        m_nodes[node_i].value = nullptr;
        for (int child_i = 0; child_i < 2; ++child_i) {
            int child_idx = m_nodes[node_i].children[child_i];
            split_leaf_recurse(child_idx, split_threshold);
        }
    }

   private:
    std::vector<Node> m_nodes;
    std::atomic<uint16_t> m_nodes_end;

    AABB m_aabb;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(STree, m_nodes, m_aabb);
};

} // namespace sdmm::accelerators

namespace nlohmann {
template <typename Scalar, int Size, typename Value>
struct adl_serializer<sdmm::accelerators::STreeNode<Scalar, Size, Value>> {
    using STreeNode = sdmm::accelerators::STreeNode<Scalar, Size, Value>;
    static void to_json(json& j, const STreeNode& node) {
        j = json{
            {"is_leaf", node.is_leaf},
            {"axis", node.axis},
            {"idx", node.idx},
            {"children", node.children},
            {"value", node.value},
            {"aabb", node.aabb},
            {"depth", node.depth}};
    }

    static STreeNode from_json(const json& j) {
        STreeNode node;
        j.at("is_leaf").get_to(node.is_leaf);
        j.at("axis").get_to(node.axis);
        j.at("idx").get_to(node.idx);
        j.at("children").get_to(node.children);
        node.value = j.at("value").get<std::unique_ptr<Value>>();
        j.at("aabb").get_to(node.aabb);
        j.at("depth").get_to(node.depth);
        return node;
    }
};
} // namespace nlohmann
