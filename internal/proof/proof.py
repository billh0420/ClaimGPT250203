# proof.py

from internal.utility.vprint import vprint


class Proof:

    def __init__(self, frame_stack, labels):
        self.frame_stack = frame_stack
        self.labels = labels

    def decompress_proof(self, stat, proof):
        dm, mand_hyp_stmts, hyp_stmts, stat = self.frame_stack.make_assertion(stat)

        mand_hyps = [self.frame_stack.lookup_f(v) for k, v in mand_hyp_stmts]
        hyps = [self.frame_stack.lookup_e(s) for s in hyp_stmts]

        labels = mand_hyps + hyps
        hyp_end = len(labels)
        ep = proof.index(')')
        labels += proof[1:ep]
        compressed_proof = ''.join(proof[ep+1:])

        vprint(5, 'labels:', labels)
        vprint(5, 'proof:', compressed_proof)

        proof_ints = []
        cur_int = 0

        for ch in compressed_proof:
            if ch == 'Z':
                proof_ints.append(-1)
            elif 'A' <= ch <= 'T':
                cur_int = (20*cur_int + ord(ch) - ord('A') + 1)
                proof_ints.append(cur_int - 1)
                cur_int = 0
            elif 'U' <= ch <= 'Y':
                cur_int = (5*cur_int + ord(ch) - ord('U') + 1)
        vprint(5, 'proof_ints:', proof_ints)

        label_end = len(labels)
        decompressed_ints = []
        subproofs = []
        prev_proofs = []
        for pf_int in proof_ints:
            if pf_int == -1:
                subproofs.append(prev_proofs[-1])
            elif 0 <= pf_int < hyp_end:
                prev_proofs.append([pf_int])
                decompressed_ints.append(pf_int)
            elif hyp_end <= pf_int < label_end:
                decompressed_ints.append(pf_int)
                step = self.labels[labels[pf_int]]
                step_type, step_data = step[0], step[1]
                if step_type in ('$a', '$p'):
                    sd, svars, shyps, sresult = step_data
                    nshyps = len(shyps) + len(svars)
                    if nshyps != 0:
                        new_prevpf = [s for p in prev_proofs[-nshyps:]
                                      for s in p] + [pf_int]
                        prev_proofs = prev_proofs[:-nshyps]
                        vprint(5, 'nshyps:', nshyps)
                    else:
                        new_prevpf = [pf_int]
                    prev_proofs.append(new_prevpf)
                else:
                    prev_proofs.append([pf_int])
            elif label_end <= pf_int:
                pf = subproofs[pf_int - label_end]
                vprint(5, 'expanded subpf:', pf)
                decompressed_ints += pf
                prev_proofs.append(pf)
        vprint(5, 'decompressed ints:', decompressed_ints)

        return [labels[i] for i in decompressed_ints]

    @staticmethod
    def decompress_proof_2(statement_label, stat, proof, frame_stack, parser_labels):
        dm, mand_hyp_stmts, hyp_stmts, stat = frame_stack.make_assertion(stat)

        mand_hyps = [frame_stack.lookup_f(v) for k, v in mand_hyp_stmts]
        hyps = [frame_stack.lookup_e(s) for s in hyp_stmts]

        labels = mand_hyps + hyps
        hyp_end = len(labels)
        ep = proof.index(')')
        labels += proof[1:ep]
        compressed_proof = ''.join(proof[ep+1:])

        vprint(5, 'labels:', labels)
        vprint(5, 'proof:', compressed_proof)

        proof_ints = []
        cur_int = 0

        for ch in compressed_proof:
            if ch == 'Z':
                proof_ints.append(-1)
            elif 'A' <= ch <= 'T':
                cur_int = (20*cur_int + ord(ch) - ord('A') + 1)
                proof_ints.append(cur_int - 1)
                cur_int = 0
            elif 'U' <= ch <= 'Y':
                cur_int = (5*cur_int + ord(ch) - ord('U') + 1)
        vprint(5, 'proof_ints:', proof_ints)

        label_end = len(labels)
        decompressed_ints = []
        subproofs = []
        prev_proofs = []
        for pf_int in proof_ints:
            if pf_int == -1:
                subproofs.append(prev_proofs[-1])
            elif 0 <= pf_int < hyp_end:
                prev_proofs.append([pf_int])
                decompressed_ints.append(pf_int)
            elif hyp_end <= pf_int < label_end:
                decompressed_ints.append(pf_int)
                step = parser_labels[labels[pf_int]]
                step_type, step_data = step[0], step[1]
                if step_type in ('$a', '$p'):
                    sd, svars, shyps, sresult = step_data
                    nshyps = len(shyps) + len(svars)
                    if nshyps != 0:
                        new_prevpf = [s for p in prev_proofs[-nshyps:]
                                      for s in p] + [pf_int]
                        prev_proofs = prev_proofs[:-nshyps]
                        vprint(5, 'nshyps:', nshyps)
                    else:
                        new_prevpf = [pf_int]
                    prev_proofs.append(new_prevpf)
                else:
                    prev_proofs.append([pf_int])
            elif label_end <= pf_int:
                pf = subproofs[pf_int - label_end]
                vprint(5, 'expanded subpf:', pf)
                decompressed_ints += pf
                prev_proofs.append(pf)
        vprint(5, 'decompressed ints:', decompressed_ints)

        return [labels[i] for i in decompressed_ints]
