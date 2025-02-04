# create_claim_corpus.py
# from claim_gpt

import re

from pathlib import Path

from corpus_base.get_mmx_file_path import get_mmx_file_path
from corpus_base.corpus01.get_corpus01_file_path import get_corpus01_file_path

from shared import Parser03 as Parser

def create_claim_corpus(corpus_file_path: Path):
    # generate_all_claims(limit_count=10)
    all_utterances = _generate_all_claims(limit_count=None)
    print(f'=== some claim_utterances ===')
    claim_utterances = []
    for i, utterance in enumerate(all_utterances):
        if utterance[0] == 'claim':
            claim_utterance = []
            ref = utterance[1]
            claim_utterance.append(ref)
            claim_utterance.append('<|start_claim|>')
            for premise in utterance[2]:
                claim_utterance.append('<|given|>')
                claim_utterance += premise.split()
            claim_utterance.append('<|conclude|>')
            claim_utterance += utterance[3].split()
            claim_utterance.append('<|end_claim|>')
            claim_utterances.append(claim_utterance)
            if i < 10:
                print(' '.join(claim_utterance))
    claim_corpus_file_name = 'claim_corpus.txt'
    claim_corpus_file_path = corpus_file_path.parent.joinpath(claim_corpus_file_name)
    with open(claim_corpus_file_path, 'w') as file:
        for claim_utterance in claim_utterances:
            line = ' '.join(claim_utterance)
            file.write(f'{line}\n')

def _generate_all_claims(limit_count: int or None):
    all_utterances = []
    wffs = _get_wffs()
    mmx_file_path = get_mmx_file_path()
    parser = Parser(mmx_file_path=mmx_file_path, limit_count=limit_count)
    count = 0
    max_count = len(parser.result.items())
    proof_count = 0
    for statement_label, value in parser.result.items():
        if value[2].startswith('$a'):
            continue
        # givens = value[1].removeprefix('$e ').split(" $e ")
        givens = value[1]
        if value[2].startswith('$p'):
            parts = re.split(r" \$= ", value[2].removesuffix(' $.'))
            proof = parts[1].split()
            # proof = [x for x in proof if x not in wffs]
        else:
            proof = None
        if proof:
            proof_count += 1
            utterances = _get_claim_utterances(proof, givens, wffs, parser)
            all_utterances += utterances
            if count < 10:
                print(f'\n=== {statement_label} ===')
                print(f'conclusion: {value[2]}')
                for utterance in utterances:
                    print(f'utterance={utterance}')
        count += 1
        if count % 1000 == 0:
            print(f'generate_all_claims2: count={count} of {max_count} proof_count={proof_count} #all_utterances={len(all_utterances)}')
    print(f'#all_utterances={len(all_utterances)}')
    return all_utterances

def _get_wffs():
    wffs = dict()
    corpus01_file_path = get_corpus01_file_path()
    file = open(corpus01_file_path, 'r')
    while True:
        statement = file.readline()
        if not statement:
            break
        tokens = statement.rstrip().split(" ")
        if len(tokens) > 4 and tokens[0] == '$f' and tokens[2] == 'wff':
            label = tokens[1]
            expression = ' '.join(tokens[3:-1])
            wffs[label] = expression
    file.close()
    return wffs

def _get_claim_utterances(proof, hypotheses, wffs, parser):
    utterances = []
    givens = dict()
    stack = []
    for hyp in hypotheses:
        tau = hyp.split(' ', 1)
        givens[tau[0]] = tau[1]
    for item in proof:
        if item in givens:
            stack.append(givens[item])
        elif item in wffs:
            stack.append(f'wff {wffs[item]}')
        elif item in parser.labels and parser.labels[item][0] == '$f':
            stack.append(' '.join(parser.labels[item][1]))
        else:
            alpha = parser.result[item]
            if alpha[2].startswith('$a') or alpha[2].startswith('$p'):
                f_hyps = alpha[0]
                e_hyps = alpha[1]
                hyps = f_hyps + e_hyps
                prop = alpha[2].removesuffix(' $.').split(' ', 2)[-1]
                if alpha[2].startswith('$p'):
                    prop = prop.split(' $= ', 1)[0]
                target_f_hyps = stack[-len(hyps): len(stack) - len(e_hyps)]
                if len(e_hyps) > 0:
                    target_e_hyps = stack[-len(e_hyps):]
                else:
                    target_e_hyps = []
                subst = dict()
                for f_hyp, target_f_hyp in zip(f_hyps, target_f_hyps):
                    subst[f_hyp.split(' ', 1)[1]] = target_f_hyp.split(' ', 1)[1]
                y = " ".join([subst.get(x, x) for x in prop.split()])
                stack = stack[:len(stack)-len(hyps)] + [y]
                if prop.startswith('|-'):
                    utterances.append(['claim', item, target_e_hyps, y])
    return utterances
