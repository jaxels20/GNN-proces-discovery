from project.evaluation.petrinets import PetrinetEvaluation
from project.data_handling.petrinet import PetrinetHandler
from pm4py.objects.petri import utils
import os
import numpy as np

ours_filename = '/home/dominique/TUe/thesis/git_data/process_trees_medium_ws2/logs_compressed/predictions/<dataset>_gcn_sound.pnml'
others_filename = '/home/dominique/TUe/thesis/git_data/process_trees_medium_ws2/logs/predictions/<dataset>_<method>.pnml'
ground_truth_filename = '/home/dominique/TUe/thesis/git_data/process_trees_medium_ws2/petrinets/<dataset>.pnml'

sound_indices = [0, 2, 4, 9, 10, 11, 13, 14, 15, 24, 25, 26, 27, 28, 30, 31, 36, 39, 53, 59, 64, 75, 77, 80, 83, 84, 88, 89, 91, 93, 97, 100, 102, 104, 106, 108, 112, 114, 116, 119, 125, 127, 136, 138, 141, 144, 145, 151, 152, 156, 165, 166, 167, 169, 179, 180, 181, 191, 192, 193, 194, 196, 199, 200, 202, 204, 205, 206, 210, 220, 226, 227, 231, 232, 233, 236, 242, 253, 257, 266, 268, 269, 279, 280, 301, 305, 306, 312, 313, 315, 320, 324, 325, 326, 328, 329, 341, 358, 359, 363, 369, 370, 372, 375, 377, 380, 385, 387, 388, 390, 391, 393, 396, 400, 406, 414, 417, 438, 440, 457, 458, 459, 465, 466, 474, 476, 478, 480, 488, 496, 501, 503, 504, 507, 518, 520, 525, 528, 532, 533, 536, 541, 547, 552, 553, 554, 559, 560, 565, 566, 568, 569, 573, 576, 577, 578, 582, 586, 591, 594, 598, 600, 601, 603, 604, 613, 625, 626, 630, 633, 638, 639, 641, 651, 657, 662]
easy_sound_indices = [0, 1, 2, 4, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 24, 25, 26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 39, 40, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 64, 73, 75, 76, 77, 78, 80, 81, 83, 84, 87, 88, 89, 91, 93, 97, 100, 101, 102, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116, 119, 120, 121, 124, 125, 127, 128, 129, 130, 131, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 155, 156, 158, 161, 162, 164, 165, 166, 167, 169, 172, 173, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 201, 202, 203, 204, 205, 206, 209, 210, 211, 218, 219, 220, 221, 223, 224, 226, 227, 228, 229, 231, 232, 233, 235, 236, 237, 239, 241, 242, 244, 245, 250, 253, 255, 257, 259, 260, 261, 262, 264, 265, 266, 268, 269, 272, 274, 278, 279, 280, 281, 285, 286, 287, 288, 290, 291, 296, 297, 300, 301, 302, 303, 304, 305, 306, 309, 310, 312, 313, 314, 315, 316, 318, 319, 320, 321, 322, 324, 325, 326, 328, 329, 331, 332, 333, 334, 336, 339, 341, 342, 344, 345, 347, 348, 350, 352, 353, 354, 357, 358, 359, 360, 361, 363, 366, 368, 369, 370, 372, 375, 377, 378, 379, 380, 382, 384, 385, 387, 388, 390, 391, 392, 393, 396, 400, 401, 402, 403, 404, 406, 407, 409, 414, 417, 418, 419, 420, 421, 422, 423, 432, 433, 434, 435, 437, 438, 439, 440, 441, 442, 443, 444, 445, 448, 449, 450, 451, 452, 455, 456, 457, 458, 459, 460, 465, 466, 467, 468, 471, 472, 473, 474, 475, 476, 478, 480, 482, 483, 484, 486, 488, 489, 490, 491, 493, 496, 497, 498, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 513, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 528, 529, 530, 531, 532, 533, 534, 535, 536, 538, 541, 542, 543, 544, 545, 546, 547, 548, 549, 551, 552, 553, 554, 556, 557, 558, 559, 560, 564, 565, 566, 568, 569, 570, 571, 572, 573, 575, 576, 577, 578, 580, 581, 582, 585, 586, 587, 588, 589, 591, 592, 594, 596, 598, 600, 601, 602, 603, 604, 606, 607, 608, 609, 610, 612, 613, 614, 619, 620, 622, 625, 626, 628, 630, 633, 634, 635, 637, 638, 639, 640, 641, 642, 645, 646, 649, 651, 653, 657, 658, 660, 661, 662]

def fixie(handler):
  init_place, final_place = None, None
  transitions_to_remove = []
  init_places = []
  for init_place in handler.mInitialMarking:
    for arc in init_place.out_arcs:
      transitions_to_remove.append(arc.target)
      init_places.extend([arc2.target for arc2 in arc.target.out_arcs])
  if init_place is not None:
    del handler.mInitialMarking[init_place]
    utils.remove_place(handler.mPetrinet, init_place)
  for init_place2 in init_places:
    handler.mInitialMarking[init_place2] += 1
  final_places = []
  for final_place in handler.mFinalMarking:
    for arc in final_place.in_arcs:
      transitions_to_remove.append(arc.source)
      final_places.extend([arc2.source for arc2 in arc.source.in_arcs])
  if final_place is not None:
    del handler.mFinalMarking[final_place]
    utils.remove_place(handler.mPetrinet, final_place)

  for final_place2 in final_places:
    handler.mFinalMarking[final_place2] += 1
  for t in transitions_to_remove:
    utils.remove_transition(handler.mPetrinet, t)


datasets = list(range(2000, 2663))
for method in ['ours', 'split', 'heuristics', 'inductive']:
  total = 0
  correct_places = 0
  total_candidates = 0
  correct_candidates = 0
  percentages = []
  print(method)
  for index, dataset in enumerate(datasets):
    if index not in easy_sound_indices:
      continue
    if method == 'ours':
      ours_pnml = f'{dataset:04d}'.join(ours_filename.split('<dataset>'))
    else:
      ours_pnml = f'{dataset:04d}'.join(others_filename.split('<dataset>'))
      ours_pnml = f'{method}'.join(ours_pnml.split('<method>'))

    gt_pnml = f'{dataset:04d}'.join(ground_truth_filename.split('<dataset>'))
    if os.path.exists(ours_pnml):
      total += 1
      ours = PetrinetHandler()
      ours.importFromFile(ours_pnml)

      if method == 'ours':
        fixie(ours)

      gt = PetrinetHandler()
      gt.importFromFile(gt_pnml)

      evaluator = PetrinetEvaluation(ours.mPetrinet, None, None)
      compared = evaluator.compare(gt.mPetrinet, None, None)
      total_candidates += compared[2]
      correct_candidates += compared[1]
      percentages.append(compared[1] / compared[2])
      if compared[0]:
        correct_places += 1
      # else:
      #   ours.visualize()
      #   gt.visualize()
      #   break

  print(correct_places / total * 100)
  print(np.mean(percentages) * 100)
  print(correct_candidates / total_candidates * 100)
