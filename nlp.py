# -*- coding: utf-8 -*-
import numpy as np
import spacy
from spacy import displacy

from Bert import modeling
from Bert.tokenization import BasicTokenizer, WordpieceTokenizer, load_vocab
from hparams import HParams
from settings import SYSTEM_ROOT, BERT_PARAMS_FILE, BERT_CONFIG_FILE, VOCAB_FILE
from settings import SUBJECT_LIST, OBJECT_LIST, PERSON_PRONOUN, NOUN_LIST, PRON_DICT, VOCAB_DICT, STOPWORDS_LIST

class Tokenizer(object):
    """Runs end-to-end tokenziation."""
    """
    Rebuild from bert.tokenization.
    Change mode [ Basic / Full ] to different tokenize that you want.
    Set unknown token and id in hparams.\
    """
    def __init__(self, hparams, vocab_file):
        self.hparams = hparams
        self.mode = hparams.tokenize_mode
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=hparams.do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
    
    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            if self.mode=="Full":
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            elif self.mode=="Basic":
                split_tokens.append(token)
        return split_tokens
    
    def de_subtokenize(self, subtokens):
        tokens = []
        for index, token in enumerate(subtokens):
            if token.startswith("##"):
                if tokens:
                    tokens[-1] = "{}{}".format(tokens[-1], token[2:])
            else:
                tokens.append(token)
        return tokens
                
    def convert_by_vocab(self, vocab, items, unknown):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            try:
                output.append(vocab[item])
            except:
                output.append(vocab[unknown])
        return output
    
    def convert_tokens_to_ids_single(self, token):
        try:
            return self.vocab[token]
        except:
            return self.vocab[self.hparams.unk_token]
    
    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens, self.hparams.unk_token)
    
    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids, self.hparams.unk_id)

class Relation_Extraction():
    def __init__(self, model="en"):
        print("# Load spacy model '%s'" % model)
        self.nlp = spacy.load(model)
        
        self.vocab_dict = VOCAB_DICT
        self.inv_vocab_dict = {v: k for k, v in self.vocab_dict.items()}
    
    def split_sentence(self, sentence):
        span = self.nlp(sentence)[:]
        tokens = [w.text for w in span]
        sentence = " ".join(tokens)
        return sentence, tokens
    
    def get_triples(self, sentence):
        sentence = self.clear_stopwords_sentence(sentence)
        doc = self.nlp(sentence)
        return self.extract_relations(doc)
    
    def build_triples_sentence(self, relation):
        sentence, triples = relation
        _, tokens = self.split_sentence(sentence)
        # if this sentence have triples
        if triples:
            sentence = ""
            for triple in triples:
                for key, value in triple.items():
                    sentence += "{} {} ".format(self.inv_vocab_dict[key], value)
        # if this sentence very long but don't have triples
        elif len(tokens) >= 7:
            sentence = self.inv_vocab_dict['random']
            for index, choose in enumerate(np.random.randint(2, size = len(tokens))):
                if choose:
                    sentence += " {}".format(tokens[index])
        # if this sentence is short
        else:
            return self.inv_vocab_dict['greeting']
        return sentence
    
    def relabel_entity(self, doc, span, ent_list):
        for ent in ent_list:
            if span.start <= ent.start and ent.end <= span.end:
                return doc.char_span(span.start_char, span.end_char, label=ent.label)
        return span
    
    def get_relation_part(self, doc):
        relation_part = {}
        for word in doc:
            if word.dep_ == "prep":
                if "pobj" in [w.dep_ for w in word.children] and "dobj" in [w.dep_ for w in word.head.children]:
                    pobj = [w for w in word.children if w.dep_ == "pobj"][0]
                    relation_part.update({word:[word, pobj]})
                elif "pobj" in [w.dep_ for w in word.children] and word.head.dep_ == "attr":
                    pobj = [w for w in word.children if w.dep_ == "pobj"][0]
                    relation_part.update({word:[word, pobj]})
                else:
                    relation_part.update({word:[word.head, word]})
            elif word.dep_ == "pcomp" and word.head.dep_ == "prep":
                relation_part.update({word:[word.head.head , word.head, word]})
            elif word.dep_ in ["acomp", "attr"] or word.pos_ == "VERB":
                relation_part.update({word:[word]})
        return relation_part
    
    def find_pronoun(self, doc, word):
        for key, value in PRON_DICT.items():
            if word.text.lower() in value:
                pron_type = key
                for w in reversed(doc[:word.i]):
                    if w.pos_ in NOUN_LIST and w.ent_type_ == pron_type:
                        return w, w.ent_type_
        return word, word.ent_type_
    
    def explain_pronoun(self, doc, relation_dict):
        try:
            if relation_dict['object'].pos_ == "PRON":
                relation_dict['object'], relation_dict['object_type'] = self.find_pronoun(doc, relation_dict['object'])
        except:
            pass
        try:
            if relation_dict['subject'].pos_ == "PRON":
                relation_dict['subject'], relation_dict['subject_type'] = self.find_pronoun(doc, relation_dict['subject'])
        except:
                pass
    
    def get_relations(self, doc, relation_part):
        relations = []
        relation_dict = {}
        for word in doc:
            if relation_part.get(word):
                for r in relation_part[word]:
                    # is 
                    if r.dep_ in ["attr", "acomp"]:
                        for w in r.head.children:
                            if w.dep_ in SUBJECT_LIST:
                                relation_dict['subject'] = w
                                relation_dict['subject_type'] = w.ent_type_
                        relation_dict['object'] = r
                        relation_dict['object_type'] = r.ent_type_ if r.ent_type_ else r.pos_
                        if relation_dict.get('object') and relation_dict.get('subject'):
                            self.explain_pronoun(doc, relation_dict)
                            # relation_dict['relation'] = r.head.text
                            relation_dict['relation'] = r.head.lemma_
                            relations.append(relation_dict)
                            relation_dict = {}
                    elif r.dep_ == "pobj":
                        if r.head.dep_ == "prep" and r.head.head.pos_ == "VERB":
                            obj = [w for w in r.head.head.children if w.dep_ in OBJECT_LIST]
                            relation_dict['subject'] = obj[0] if obj else obj
                            relation_dict['subject_type'] = obj[0].ent_type_ if obj else obj
                            relation_dict['object'] = r
                            relation_dict['object_type'] = r.ent_type_
                            if relation_dict.get('object') and relation_dict.get('subject'):
                                self.explain_pronoun(doc, relation_dict)
                                relation_dict['relation'] = " ".join(["by", r.head.head.lemma_, r.head.lemma_])
                                # relation_dict['relation'] = " ".join(["by", r.head.head.text, r.head.text])
                                relations.append(relation_dict)
                                relation_dict = {}
                        elif r.head.dep_ == "prep" and r.head.head.dep_ == "attr":
                            subj = [w for w in r.head.head.head.children if w.dep_ in SUBJECT_LIST]
                            relation_dict['subject'] = subj[0] if subj else subj
                            relation_dict['subject_type'] = subj[0].ent_type_ if subj else subj
                            # relation_dict['object'] = " ".join([r.head.head.text]+[w.text for w in relation_part[word]])
                            relation_dict['object'] = " ".join([r.head.head.lemma_]+[w.lemma_ for w in relation_part[word]])
                            relation_dict['object_type'] = r.head.head.ent_type_
                            if relation_dict.get('object') and relation_dict.get('subject'):
                                self.explain_pronoun(doc, relation_dict)
                                relation_dict['relation'] = r.head.head.head.lemma_
                                # relation_dict['relation'] = r.head.head.head.text
                                relations.append(relation_dict)
                                relation_dict = {}
                    # VERB
                    else:
                        for w in r.children:
                            if w.dep_ in SUBJECT_LIST:
                                relation_dict['subject'] = w
                                relation_dict['subject_type'] = w.ent_type_
                            elif w.dep_ in OBJECT_LIST:
                                relation_dict['object'] = w
                                relation_dict['object_type'] = w.ent_type_
                        if relation_dict.get('object') and relation_dict.get('subject'):
                            self.explain_pronoun(doc, relation_dict)
                            relation_dict['relation'] = " ".join([w.text for w in relation_part[word]])
                            relations.append(relation_dict)
                            relation_dict = {}
        return relations

    def open_conjuncts(self, relations):
        # remix conjunct word
        new_relations = []
        for relation in relations:
            try:
                if relation['subject'].conjuncts:
                    for sub_conjunct in (list(relation['subject'].conjuncts) + [relation['subject']]):
                        relation['subject'] = sub_conjunct
                        relation['subject_type'] = sub_conjunct.ent_type_
                        try:
                            if relation['object'].conjuncts:
                                for obj_conjunct in (list(relation['object'].conjuncts) + [relation['object']]):
                                    relation['object'] = obj_conjunct
                                    relation['object_type'] = obj_conjunct.ent_type_
                                    new_relations.append(relation.copy())
                            else:
                                new_relations.append(relation.copy())
                        except:
                            new_relations.append(relation.copy())
                elif relation['object'].conjuncts:
                    for obj_conjunct in (list(relation['object'].conjuncts) + [relation['object']]):
                        relation['object'] = obj_conjunct
                        relation['object_type'] = obj_conjunct.ent_type_
                        new_relations.append(relation.copy())
                else:
                    new_relations.append(relation)
            except:
                new_relations.append(relation)
        return new_relations
    
    def clear_stopwords_sentence(self, sentence):
        sentence = ' '.join(["" if word in STOPWORDS_LIST else word for word in sentence.split(" ")])
        return sentence
    
    def clear_stopwords_relation(self, relations):
        if type(relations) == type(list()):
            for relation in relations:
                for key, value in relation.items():
                    relation[key] = self.clear_stopwords_sentence(str(value))
    
    def extract_relations(self, doc):
        with doc.retokenize() as retokenizer:
            entity_spans = spacy.util.filter_spans(list(doc.ents) + list(doc.noun_chunks))
            for span in entity_spans:
                span = self.relabel_entity(doc, span, list(doc.ents))
                retokenizer.merge(span, attrs = {'ent_type': span.label})
                
        # displacy.serve(doc, style="dep")
        
        relation_part = self.get_relation_part(doc)
        relations = self.get_relations(doc, relation_part)
        relations = self.open_conjuncts(relations)
        self.clear_stopwords_relation(relations)
        return (doc.text, relations)
    
if __name__ == "__main__":
    # TEXTS = ["Apple is looking at buying U.K. startup for $1 billion.",
    #           "Autonomous cars shift insurance liability toward manufacturers.",
    #           "Barrack Obama was born in Hawaii in the year 1961. He was president of the United States.",
    #           "Apple was founded in Cupertino in the year 1981.",
    #           "Have you heard of Obama? He is the president of the united states",
    #           "I like apple and banana, they are very tasty."]
    
    # TEXTS = ["Not the hacking and gagging and spitting part . Please .",
    #          "The thing is , Cameron -- I 'm at the mercy of a particularly hideous breed of loser . My sister . I ca n't date until she does .",
    #         "Me . This endless ... blonde babble . I 'm like , boring myself .",
    #         "Could you help me make a plane reservation ?",
    #         "I would be happy to help you . Where do you plan on going ?",
    #         "I am going to go to Hawaii ."]
    
    extracter = Relation_Extraction()
    
    for test in TEXTS:
        output = extracter.get_triples(test.lower())
        print(output)
        
        output = extracter.build_triples_sentence(output)
        print(output)
    