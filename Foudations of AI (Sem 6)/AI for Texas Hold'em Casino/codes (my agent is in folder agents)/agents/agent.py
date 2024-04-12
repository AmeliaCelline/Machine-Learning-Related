from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator
import copy
import random


class CallPlayer(
    BasePokerPlayer
):  
    def __init__(self):
        self.suit_map = ["C", "D", "H", "S"]
        self.rank_map = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

        self.round = 0
        self.smallblind = 0
        self.start = 0 
        self.total_raise = 0
        self.fold = 0
        self.chips = 0
    
    
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        if self.fold == 1:
            return valid_actions[0]['action'], valid_actions[0]['amount'] 
            
        temp = int(self.round/2)
        temp2 = int(self.round%2)
        
        total = temp * self.smallblind + (temp+temp2) * 2 * self.smallblind
        
        if (valid_actions[2]['amount']['max'] - total > self.start):
            self.fold = 1
            
            return valid_actions[0]['action'], valid_actions[0]['amount'] 
        
        #get our initial chip before round start
        if round_state['street'] == 'preflop':
            self.chips = valid_actions[2]['amount']['max']
        
        
        #for monte carlo
        n_simulation = 5000
        remainder_c = 5 - len(round_state['community_card'])
        
        winning = 0

        converted_hole_card = []
        for i in hole_card:
            converted_hole_card.append(Card.from_str(i))
        
        converted_community_card = []
        for i in round_state['community_card']:
            converted_community_card.append(Card.from_str(i))
            
            
        #monte carlo simulation
        for i in range(n_simulation):
            #cur_community is the community card in the simulation
            cur_community = copy.deepcopy(round_state['community_card'])
            #opponent is the opponent card in the simulation
            opponent = []
            
            copy_remainder_c = remainder_c
            opponent_card = 2
            
            while True:
                s = random.choice(self.suit_map)
                r = random.choice(self.rank_map)
                
                card = s + r
                if card not in cur_community and card not in hole_card and card not in opponent:
                
                    if copy_remainder_c > 0:
                        cur_community.append(card)
                        copy_remainder_c -= 1
                    elif opponent_card > 0:
                        opponent.append(card)
                        opponent_card -= 1
                    else:
                        break
                
            converted_cur_community = copy.deepcopy(converted_community_card)
            converted_opponent = []
            for j in range(remainder_c):
                converted_cur_community.append(Card.from_str(cur_community[4-j]))
            for j in range(2):
                converted_opponent.append(Card.from_str(opponent[j]))
            
            
            #evaluate the strength
            agent_strength = HandEvaluator.eval_hand(converted_hole_card, converted_cur_community)
            opponent_strength = HandEvaluator.eval_hand(converted_opponent, converted_cur_community)
            
            if agent_strength >= opponent_strength:
                winning += 1
                
        threshold_top = self.chips - round_state['pot']['main']['amount']
        prob = winning/n_simulation
        good_card = 0
        if round_state['street'] == 'preflop':
            
            rank_card0 = converted_hole_card[0].rank
            rank_card1 = converted_hole_card[1].rank
            if (hole_card[0][1] == hole_card[1][1]):
                good_card = 1
            elif (rank_card1 >= 11 or rank_card0 >= 11) or (abs(rank_card0 - rank_card1) < 5):
                good_card = 1
            
            if good_card == 0 and valid_actions[1]['amount'] > 2 * self.smallblind:
                prob *= 0.5
            
        elif prob < 0.8 and valid_actions[1]['amount']> 0.03 * self.chips and good_card == 0:
            #reduce our confident
            prob *= 0.5
                    

        if prob > 0.9:
            multiplier = 0.5
            
            amt = prob * multiplier * (valid_actions[2]['amount']['max']/2)
            amt = max(valid_actions[2]['amount']['min'], min(amt, valid_actions[2]['amount']['max']))
            
            if amt > valid_actions[1]['amount'] and amt <= threshold_top:
                return valid_actions[2]['action'], amt  
            else:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
        elif prob > 0.7:
            if round_state['street'] == 'turn' or  round_state['street'] == 'river':
                multiplier = 0.3
            else:
                multiplier = 0.15 
            
            amt = prob * multiplier * (valid_actions[2]['amount']['max']/2)
            #making sure it is within the range
            amt = max(valid_actions[2]['amount']['min'], min(amt, valid_actions[2]['amount']['max']))
            
            if amt > valid_actions[1]['amount'] and amt <= threshold_top:
                return valid_actions[2]['action'], amt
            else:
                return valid_actions[1]['action'], valid_actions[1]['amount']

        elif prob > 0.4:
            if prob >= 0.5 or round_state['street'] == 'river': 
                multiplier = 0.07
            else:
                multiplier = 0.03
                
            if valid_actions[1]['amount'] <= multiplier * self.chips:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']
            
        else:
            if round_state['street'] == 'preflop' and valid_actions[1]['amount'] <= 0.1 * valid_actions[2]['amount']['max'] and (prob > 0.25 or good_card == 1):
                return valid_actions[1]['action'], valid_actions[1]['amount']
            elif valid_actions[1]['amount'] == 0:
                return valid_actions[1]['action'], valid_actions[1]['amount']
        
            return valid_actions[0]['action'], valid_actions[0]['amount']
        



    def receive_game_start_message(self, game_info):
        self.smallblind = game_info['rule']['small_blind_amount']
        self.start = game_info['rule']['initial_stack']
        self.round = game_info['rule']['max_round']
        self.fold = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.round -= 1
        self.total_raise = 0


def setup_ai():
    return CallPlayer()
