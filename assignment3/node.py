from graphviz import Digraph


class Node(object):
    def __init__(self, tag="", word="", children=None):
        self.tag = tag
        if children is not None:
            self.children = children
        else:
            self.children = []
        self.order = -1
        self.span = (-1, -1)
        self.word = word

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.linearize()

    def slinearize(self):
        return "({} {})".format(
        self.tag, " ".join(child.slinearize() for child in self.children))

    def blinearize(self):
        return "({} {})".format(
        self.tag, " ".join([self.left.blinearize(), self.right.blinearize()]))

    def linearize(self):
        if len(self.children) != 0:
            # print(self.children)
            return "{}".format(
                " ".join(child.linearize() for child in self.children))
        else:
            return self.tag

    def convert2CNF(self, order=1):
        if len(self.children) > 2:
            # only retain the original father node
            if order == 1:
                new_node = Node(tag="{}@{}".format(self.tag.split("@")[0] 
                                                    if "@" in self.tag else self.tag, self.children[0].tag))
            else:
                new_node = Node(tag="{}@".format(self.tag.strip("@")))
            for i in range(1, len(self.children)):
                new_node.add_child(self.children[i])

            self.children = [self.children[0], new_node]

        if len(self.children) > 0:
            self.children[0].convert2CNF(order=order)
        if len(self.children) > 1:
            self.children[1].convert2CNF(order=order)

    def order_print(self):
        if len(self.children) != 0:
            ret = []
            for child in self.children:
                ret += child.order_print()
            return ret
        else:
            return [self.order]
    
    def get_all_spans(self):
        if hasattr(self, 'word'):
            if self.word != "":
                return []
        else:
            if len(self.children) == 0:
                return []
        
        if len(self.children) > 0:
            if len(self.children) == 1 and self.children[0].word != "":
                return []
            ret = [self.span]
            for child in self.children:
                ret += child.get_all_spans() 
            return ret
        else:
            return []

    def get_original_sentence(self):
        if self.word != "":
            return self.word
        else:
            return " ".join(child.get_original_sentence() for child in self.children)

    def sen_len(self):
        return len(self.get_original_sentence().split(" "))

def from_string(s):
    tokens = s.replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(Node(label, children=children))
            else:
                word = tokens[index]
                index += 1
                trees.append(Node(label, word=word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index
    t, idx = helper(0)
    return t[0]


def draw_tree(r: Node, res_path=None):
    dot = Digraph(comment=r.linearize())
    bfs_list = [(None, r)]
    node_cnt = 0
    while len(bfs_list) > 0:
        father_node_num, curr_node = bfs_list[0]
        node_cnt += 1
        curr_node_num = node_cnt
        dot.node(str(curr_node_num), curr_node.tag)

        if father_node_num is not None:
            dot.edge(str(father_node_num), str(curr_node_num))
        if curr_node.word != "":
            node_cnt += 1
            dot.node(str(node_cnt), curr_node.word)
            dot.edge(str(curr_node_num), str(node_cnt))

        bfs_list.pop(0)

        for child in curr_node.children:
            bfs_list.append((curr_node_num, child))
    if res_path is not None:
        dot.render(res_path, format="png") 
    return dot


if __name__ == "__main__":
    with open("./data/ptb_LE10/train.pid") as f:
        for l in f:
            l = l.strip("\n")
            t = from_string(l)
            if t.sen_len() <= 10:
                print(l )
                draw_tree(t, "viz.png")
                break