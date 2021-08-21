class big_root_heap:
    def __init__(self, maxsize):
        self.heap = []
        self.maxsize = maxsize

    def len(self):
        return len(self.heap) - 1

    def adjust_heap(self, begin):
        end = self.len()
        self.heap[0] = self.heap[end]
        i = 2 * begin
        while i <= end:
            if (i < end and self.heap[i]["distance"] < self.heap[i + 1]["distance"]):
                i += 1
            if (self.heap[0]["distance"] >= self.heap[i]["distance"]):
                break
            else:
                self.heap[begin] = self.heap[i]
                begin = i
            self.heap[begin] = self.heap[0]
            i *= 2

    def build_heap(self):
        i = self.len() // 2
        while i >= 1:
            self.adjust_heap(i)
            i -= 1

    def push(self, item):
        if self.len() < self.maxsize:
            self.heap.append(item)
            self.build_heap()
        else:
            top = self.top()
            if top["distance"] > item["distance"]:
                self.pop()
                self.heap.append(item)
                self.build_heap()

    def pop(self):
        self.heap.pop(1)

    def top(self):
        return self.heap[1]


if __name__ == "__main__":
    heap = big_root_heap(maxsize=2)
    test = [15,10,7,25,31,15]
    for i,each in enumerate(test):
        heap.push({'index':i,'distance':each})

    for i, each in enumerate(heap.heap):
        if i:
            print("index: {}".format(each["index"]), end='  ')
            print("distance: {}".format(each["distance"]))