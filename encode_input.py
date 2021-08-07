class EncodeInput:
    def __init__(self,alpha):
        self.alpha = alpha
                
        self.hand_count = {
            'K': 0,
            'L': 0,
            'paper': 0,
            'rock': 0,
            'scissor': 0,
            'W': 0
        }
        self.none_count = 0

    def reset_count(self):
        self.hand_count = {
            'K': 0,
            'L': 0,
            'paper': 0,
            'rock': 0,
            'scissor': 0,
            'W': 0
        }
        self.none_count = 0

    def accumulate(self,hands):
        if len(hands) == 1:
            hand = hands[0]
            self.hand_count[hand] += 1
        
        else: 
            self.none_count += 1
        

    def verify(self):
        output = None
        if self.none_count == self.alpha:
            self.reset_count()

        elif max(self.hand_count.values()) == self.alpha:
            output = max(self.hand_count, key=self.hand_count.get)
            self.reset_count()
        
        return output

    def encode(self,hands):
        self.accumulate(hands)
        output = self.verify()
        if output == None: pass
        return output

if __name__ == '__main__':
    EI = EncodeInput(10)
    while True:
        hands = list(input().split())
        command = EI.encode(hands)
        if command: print(f'==={command}===')
