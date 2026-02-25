"""
Pygame version of Rummikub
A graphical implementation with local game logic
"""

import pygame
import sys
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

from tile import Tile, TileSet, sort_tiles_for_hand
from meld import Meld
from game_state import GameState

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60

# Colors
BACKGROUND_COLOR = (20, 80, 40)  # Dark green felt
TABLE_COLOR = (30, 100, 50)  # Lighter green for table area
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)
HIGHLIGHT_COLOR = (255, 255, 100)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 160, 210)
BUTTON_TEXT_COLOR = WHITE

# Tile Colors
TILE_RED = (220, 50, 50)
TILE_BLUE = (50, 50, 220)
TILE_BLACK = (40, 40, 40)
TILE_ORANGE = (255, 140, 0)
TILE_JOKER = (180, 100, 180)

# Tile dimensions
TILE_WIDTH = 50
TILE_HEIGHT = 70
TILE_MARGIN = 5
TILE_CORNER_RADIUS = 8

# Fonts
pygame.font.init()
font_large = pygame.font.SysFont('Arial', 24, bold=True)
font_medium = pygame.font.SysFont('Arial', 18, bold=True)
font_small = pygame.font.SysFont('Arial', 14)
font_tile_number = pygame.font.SysFont('Arial', 28, bold=True)


@dataclass
class Button:
    """Represents a clickable button"""
    x: int
    y: int
    width: int
    height: int
    text: str
    callback: callable
    enabled: bool = True
    
    def draw(self, screen: pygame.Surface, mouse_pos: Tuple[int, int]):
        """Draw the button"""
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        # Check hover
        is_hovered = rect.collidepoint(mouse_pos) and self.enabled
        color = BUTTON_HOVER_COLOR if is_hovered else BUTTON_COLOR
        
        if not self.enabled:
            color = (100, 100, 100)
        
        # Draw button background
        pygame.draw.rect(screen, color, rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, rect, 2, border_radius=5)
        
        # Draw text
        text_surface = font_medium.render(self.text, True, BUTTON_TEXT_COLOR if self.enabled else GRAY)
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """Handle click event, return True if clicked"""
        if not self.enabled:
            return False
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        if rect.collidepoint(mouse_pos):
            self.callback()
            return True
        return False


class TileRenderer:
    """Handles rendering of tiles"""
    
    @staticmethod
    def get_tile_color(tile: Tile) -> Tuple[int, int, int]:
        """Get the display color for a tile"""
        if tile.is_joker:
            return TILE_JOKER
        color_map = {
            'red': TILE_RED,
            'blue': TILE_BLUE,
            'black': TILE_BLACK,
            'orange': TILE_ORANGE
        }
        return color_map.get(tile.color, GRAY)
    
    @staticmethod
    def draw_tile(screen: pygame.Surface, tile: Tile, x: int, y: int, 
                  selected: bool = False, highlighted: bool = False):
        """Draw a single tile at the given position"""
        rect = pygame.Rect(x, y, TILE_WIDTH, TILE_HEIGHT)
        
        # Draw shadow
        shadow_rect = rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(screen, (0, 0, 0, 100), shadow_rect, border_radius=TILE_CORNER_RADIUS)
        
        # Draw tile background
        bg_color = WHITE
        if selected:
            bg_color = HIGHLIGHT_COLOR
        elif highlighted:
            bg_color = (200, 230, 255)
        
        pygame.draw.rect(screen, bg_color, rect, border_radius=TILE_CORNER_RADIUS)
        pygame.draw.rect(screen, BLACK, rect, 2, border_radius=TILE_CORNER_RADIUS)
        
        # Draw inner colored circle/oval
        inner_rect = rect.inflate(-10, -20)
        inner_rect.center = rect.center
        tile_color = TileRenderer.get_tile_color(tile)
        pygame.draw.ellipse(screen, tile_color, inner_rect)
        pygame.draw.ellipse(screen, BLACK, inner_rect, 1)
        
        # Draw number
        if tile.is_joker:
            text = font_medium.render("J", True, WHITE)
        else:
            text = font_tile_number.render(str(tile.number), True, WHITE)
        
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)
        
        # Draw small corner indicators
        if not tile.is_joker:
            small_text = font_small.render(str(tile.number), True, tile_color)
            screen.blit(small_text, (x + 4, y + 4))
            screen.blit(small_text, (x + TILE_WIDTH - 12, y + TILE_HEIGHT - 14))


class RummikubGame:
    """Main game class"""
    
    def __init__(self, num_players: int = 2):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Rummikub - Pygame Edition")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game state
        self.num_players = num_players
        self.game_state = GameState(num_players)
        self.game_state.reset()
        
        # UI State
        self.selected_tiles: Set[int] = set()  # Indices of selected tiles in hand
        self.selected_table_meld: Optional[int] = None  # Index of selected table meld
        self.message = ""
        self.message_timer = 0
        
        # Buttons
        self.buttons: List[Button] = []
        self._create_buttons()
        
        # Layout constants
        self.hand_y = SCREEN_HEIGHT - 280  # Moved up to allow for 3+ rows
        self.table_y_start = 120
        self.table_area_height = 320  # Reduced to make room for hand
        
    def _create_buttons(self):
        """Create game control buttons"""
        button_y = SCREEN_HEIGHT - 40  # Keep buttons at bottom
        
        self.buttons.append(Button(
            x=20, y=button_y, width=120, height=35,
            text="Draw Tile",
            callback=self._draw_tile
        ))
        
        self.buttons.append(Button(
            x=150, y=button_y, width=140, height=35,
            text="Play Meld",
            callback=self._play_meld
        ))
        
        self.buttons.append(Button(
            x=300, y=button_y, width=140, height=35,
            text="Add to Meld",
            callback=self._add_to_meld
        ))
        
        self.buttons.append(Button(
            x=450, y=button_y, width=120, height=35,
            text="Clear Selection",
            callback=self._clear_selection
        ))
        
        self.buttons.append(Button(
            x=580, y=button_y, width=120, height=35,
            text="Declare Out",
            callback=self._declare_out
        ))
        
        self.buttons.append(Button(
            x=710, y=button_y, width=100, height=35,
            text="Sort Hand",
            callback=self._sort_hand
        ))
        
        self.buttons.append(Button(
            x=SCREEN_WIDTH - 140, y=button_y, width=120, height=35,
            text="New Game",
            callback=self._new_game
        ))
    
    def _draw_tile(self):
        """Draw a tile from the pool"""
        player_id = self.game_state.current_player
        tile = self.game_state.draw_tile(player_id)
        if tile:
            self._show_message(f"Drew: {tile}")
            self.game_state.next_player()
        else:
            self._show_message("Pool is empty!")
    
    def _play_meld(self):
        """Play selected tiles as a new meld"""
        if not self.selected_tiles:
            self._show_message("Select tiles to play!")
            return
        
        player_id = self.game_state.current_player
        tile_indices = sorted(list(self.selected_tiles))
        
        if self.game_state.play_meld(player_id, tile_indices):
            self._show_message("Meld played successfully!")
            self.selected_tiles.clear()
            
            # Check if player won
            if self.game_state.can_go_out(player_id):
                self._declare_out()
            else:
                self.game_state.next_player()
        else:
            self._show_message("Invalid meld! Must be 3+ consecutive same-color or same-number different colors")
    
    def _add_to_meld(self):
        """Add selected tile to selected table meld"""
        if len(self.selected_tiles) != 1:
            self._show_message("Select exactly one tile from hand!")
            return
        
        if self.selected_table_meld is None:
            self._show_message("Select a table meld first!")
            return
        
        player_id = self.game_state.current_player
        tile_idx = list(self.selected_tiles)[0]
        meld_idx = self.selected_table_meld
        
        if self.game_state.add_to_meld(player_id, tile_idx, meld_idx):
            self._show_message("Tile added to meld!")
            self.selected_tiles.clear()
            self.selected_table_meld = None
            
            # Check if player won
            if self.game_state.can_go_out(player_id):
                self._declare_out()
            else:
                self.game_state.next_player()
        else:
            self._show_message("Cannot add tile to this meld!")
    
    def _declare_out(self):
        """Declare Rummikub (going out)"""
        player_id = self.game_state.current_player
        if self.game_state.declare_out(player_id):
            self._show_message(f"Player {player_id + 1} declares RUMMIKUB! Winner!")
        else:
            self._show_message("You still have tiles in hand!")
    
    def _clear_selection(self):
        """Clear tile selection"""
        self.selected_tiles.clear()
        self.selected_table_meld = None
    
    def _new_game(self):
        """Start a new game"""
        self.game_state.reset()
        self.selected_tiles.clear()
        self.selected_table_meld = None
        self._show_message("New game started!")
    
    def _sort_hand(self):
        """Manually sort the current player's hand"""
        player_id = self.game_state.current_player
        self.game_state.player_hands[player_id] = sort_tiles_for_hand(
            self.game_state.player_hands[player_id]
        )
        self._show_message("Hand sorted!")
    
    def _show_message(self, text: str, duration: int = 180):
        """Show a message for a duration (in frames)"""
        self.message = text
        self.message_timer = duration
    
    def _get_tile_rect(self, index: int, total_tiles: int) -> pygame.Rect:
        """Get the rectangle for a tile in the hand with multi-row support"""
        # Calculate how many tiles fit per row (with some margin)
        usable_width = SCREEN_WIDTH - 40  # 20px margin on each side
        tiles_per_row = usable_width // (TILE_WIDTH + TILE_MARGIN)
        
        # Calculate row and column
        row = index // tiles_per_row
        col = index % tiles_per_row
        
        # Calculate tiles in this row
        tiles_in_row = min(tiles_per_row, total_tiles - row * tiles_per_row)
        
        # Center the row
        row_width = tiles_in_row * (TILE_WIDTH + TILE_MARGIN) - TILE_MARGIN
        start_x = (SCREEN_WIDTH - row_width) // 2
        
        x = start_x + col * (TILE_WIDTH + TILE_MARGIN)
        y = self.hand_y + row * (TILE_HEIGHT + 5)  # 5px spacing between rows
        
        return pygame.Rect(x, y, TILE_WIDTH, TILE_HEIGHT)
    
    def _get_meld_rect(self, meld_idx: int, tile_idx: int, meld_lengths: List[int]) -> pygame.Rect:
        """Get the rectangle for a tile in a table meld with proper spacing"""
        margin_x = 15
        margin_y = 15
        meld_spacing = 25  # Space between melds
        
        # Calculate actual positions based on previous melds
        current_x = margin_x
        current_row = 0
        
        for i in range(meld_idx + 1):
            meld_width = meld_lengths[i] * (TILE_WIDTH + 2)
            
            # Check if this meld fits in current row
            if i < meld_idx:  # For previous melds, just advance position
                current_x += meld_width + meld_spacing
                if current_x + meld_lengths[i + 1] * (TILE_WIDTH + 2) > SCREEN_WIDTH - margin_x:
                    # Wrap to next row
                    current_x = margin_x
                    current_row += 1
            else:  # For current meld, calculate tile position
                x = current_x + tile_idx * (TILE_WIDTH + 2)
                y = self.table_y_start + current_row * (TILE_HEIGHT + margin_y)
                return pygame.Rect(x, y, TILE_WIDTH, TILE_HEIGHT)
        
        # Fallback (shouldn't reach here)
        return pygame.Rect(margin_x, self.table_y_start, TILE_WIDTH, TILE_HEIGHT)
    
    def handle_events(self):
        """Handle input events"""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_left_click(mouse_pos)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self._draw_tile()
                elif event.key == pygame.K_RETURN:
                    self._play_meld()
                elif event.key == pygame.K_c:
                    self._clear_selection()
                elif event.key == pygame.K_s:
                    self._sort_hand()
    
    def _handle_left_click(self, mouse_pos: Tuple[int, int]):
        """Handle left mouse click"""
        # Check buttons first
        for button in self.buttons:
            if button.handle_click(mouse_pos):
                return
        
        # Check hand tiles
        player_id = self.game_state.current_player
        hand = self.game_state.get_current_player_hand()
        
        for i in range(len(hand)):
            tile_rect = self._get_tile_rect(i, len(hand))
            if tile_rect.collidepoint(mouse_pos):
                if i in self.selected_tiles:
                    self.selected_tiles.remove(i)
                else:
                    self.selected_tiles.add(i)
                return
        
        # Check table melds
        meld_lengths = [len(m.get_display_tiles()) for m in self.game_state.table_melds]
        for meld_idx, meld in enumerate(self.game_state.table_melds):
            for tile_idx in range(len(meld.get_display_tiles())):
                tile_rect = self._get_meld_rect(meld_idx, tile_idx, meld_lengths)
                if tile_rect.collidepoint(mouse_pos):
                    self.selected_table_meld = meld_idx
                    return
    
    def draw(self):
        """Draw the game screen"""
        # Background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw table area
        table_rect = pygame.Rect(10, self.table_y_start - 10, SCREEN_WIDTH - 20, self.table_area_height)
        pygame.draw.rect(self.screen, TABLE_COLOR, table_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, table_rect, 3, border_radius=10)
        
        # Draw header
        self._draw_header()
        
        # Draw table melds
        self._draw_table_melds()
        
        # Draw hand area
        self._draw_hand()
        
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, mouse_pos)
        
        # Draw message
        if self.message_timer > 0:
            self._draw_message()
            self.message_timer -= 1
        
        # Draw instructions
        self._draw_instructions()
        
        pygame.display.flip()
    
    def _draw_header(self):
        """Draw the game header with status info"""
        # Title
        title = font_large.render("RUMMIKUB", True, WHITE)
        self.screen.blit(title, (20, 20))
        
        # Current player
        player_text = font_medium.render(
            f"Current Player: {self.game_state.current_player + 1}", 
            True, 
            HIGHLIGHT_COLOR if self.game_state.current_player == 0 else WHITE
        )
        self.screen.blit(player_text, (20, 55))
        
        # Pool size
        pool_text = font_medium.render(
            f"Tiles in Pool: {self.game_state.tile_pool.remaining()}",
            True,
            WHITE
        )
        self.screen.blit(pool_text, (220, 55))
        
        # Initial meld status
        initial_meld_text = font_medium.render(
            f"Initial Meld: {'✓ Done' if self.game_state.has_initial_meld[self.game_state.current_player] else '✗ Required (30+ points)'}",
            True,
            (100, 255, 100) if self.game_state.has_initial_meld[self.game_state.current_player] else (255, 200, 100)
        )
        self.screen.blit(initial_meld_text, (400, 55))
        
        # Turn count
        turn_text = font_medium.render(f"Turn: {self.game_state.turn_count}", True, WHITE)
        self.screen.blit(turn_text, (SCREEN_WIDTH - 150, 20))
        
        # Game over
        if self.game_state.game_over and self.game_state.winner is not None:
            winner_text = font_large.render(
                f"GAME OVER - Player {self.game_state.winner + 1} Wins!",
                True,
                (255, 255, 0)
            )
            text_rect = winner_text.get_rect(center=(SCREEN_WIDTH // 2, 90))
            self.screen.blit(winner_text, text_rect)
    
    def _draw_table_melds(self):
        """Draw all melds on the table"""
        if not self.game_state.table_melds:
            # Show placeholder text
            text = font_medium.render("No melds on table yet", True, (150, 150, 150))
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 300))
            self.screen.blit(text, text_rect)
            return
        
        # Pre-compute meld lengths for positioning
        meld_lengths = [len(meld.get_display_tiles()) for meld in self.game_state.table_melds]
        
        for meld_idx, meld in enumerate(self.game_state.table_melds):
            # Get tiles sorted for display (runs will have jokers in proper positions)
            display_tiles = meld.get_display_tiles()
            
            # Highlight selected meld
            if meld_idx == self.selected_table_meld:
                # Draw highlight around the whole meld
                first_rect = self._get_meld_rect(meld_idx, 0, meld_lengths)
                last_rect = self._get_meld_rect(meld_idx, len(display_tiles) - 1, meld_lengths)
                highlight_rect = pygame.Rect(
                    first_rect.x - 5,
                    first_rect.y - 5,
                    last_rect.right - first_rect.x + 10,
                    TILE_HEIGHT + 10
                )
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, highlight_rect, 3, border_radius=5)
            
            # Draw tiles in meld (using sorted display order)
            for tile_idx, tile in enumerate(display_tiles):
                rect = self._get_meld_rect(meld_idx, tile_idx, meld_lengths)
                TileRenderer.draw_tile(
                    self.screen, tile, rect.x, rect.y,
                    highlighted=(meld_idx == self.selected_table_meld)
                )
            
            # Draw meld value
            value = Meld.calculate_value(meld.tiles)
            last_rect = self._get_meld_rect(meld_idx, len(display_tiles) - 1, meld_lengths)
            value_text = font_small.render(f"={value}", True, WHITE)
            self.screen.blit(value_text, (last_rect.right + 5, last_rect.centery - 8))
    
    def _draw_hand(self):
        """Draw the current player's hand with multi-row support"""
        player_id = self.game_state.current_player
        hand = self.game_state.get_current_player_hand()
        
        # Calculate number of rows needed
        usable_width = SCREEN_WIDTH - 40
        tiles_per_row = usable_width // (TILE_WIDTH + TILE_MARGIN)
        num_rows = (len(hand) + tiles_per_row - 1) // tiles_per_row if hand else 1
        
        # Adjust hand background height based on number of rows
        row_height = TILE_HEIGHT + 5  # Match the spacing in _get_tile_rect
        hand_height = max(100, 35 + num_rows * row_height)
        
        # Hand background
        hand_bg = pygame.Rect(10, self.hand_y - 20, SCREEN_WIDTH - 20, hand_height)
        pygame.draw.rect(self.screen, DARK_GRAY, hand_bg, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, hand_bg, 2, border_radius=10)
        
        # Label
        label = font_medium.render("Your Hand (click to select):", True, WHITE)
        self.screen.blit(label, (20, self.hand_y - 15))
        
        # Draw tiles
        if not hand:
            empty_text = font_medium.render("Empty hand - click 'Declare Out' to win!", True, (100, 255, 100))
            self.screen.blit(empty_text, (SCREEN_WIDTH // 2 - 150, self.hand_y + 20))
            return
        
        for i, tile in enumerate(hand):
            rect = self._get_tile_rect(i, len(hand))
            TileRenderer.draw_tile(
                self.screen, tile, rect.x, rect.y,
                selected=(i in self.selected_tiles)
            )
        
        # Draw selection info (position below the last row)
        if self.selected_tiles:
            selected_indices = sorted(list(self.selected_tiles))
            tiles = [hand[i] for i in selected_indices]
            
            # Check if valid meld
            is_valid = Meld.is_valid(tiles)
            value = Meld.calculate_value(tiles) if is_valid else 0
            
            info_text = f"Selected: {len(self.selected_tiles)} tiles"
            if is_valid:
                info_text += f" (Valid meld: {value} pts)"
            
            info_surface = font_small.render(info_text, True, (100, 255, 100) if is_valid else (255, 200, 100))
            info_y = self.hand_y + num_rows * row_height + 10
            self.screen.blit(info_surface, (20, info_y))
    
    def _draw_message(self):
        """Draw the current message"""
        # Message background
        msg_bg = pygame.Rect(SCREEN_WIDTH // 2 - 300, SCREEN_HEIGHT // 2 - 40, 600, 80)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), msg_bg, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, msg_bg, 2, border_radius=10)
        
        # Message text
        msg_surface = font_large.render(self.message, True, WHITE)
        msg_rect = msg_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(msg_surface, msg_rect)
    
    def _draw_instructions(self):
        """Draw game instructions"""
        instructions = [
            "Controls:",
            "Click tiles to select/deselect",
            "Click table meld to select destination",
            "Space = Draw  |  Enter = Play Meld  |  C = Clear",
            "S = Sort Hand  |  Esc = Quit",
            "Hand is auto-sorted by color & number"
        ]
        
        y = SCREEN_HEIGHT - 120
        for instruction in instructions:
            text = font_small.render(instruction, True, LIGHT_GRAY)
            self.screen.blit(text, (SCREEN_WIDTH - 300, y))
            y += 18
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    print("Starting Rummikub Pygame Edition...")
    print("Controls:")
    print("  - Click tiles to select/deselect")
    print("  - Click 'Play Meld' to play selected tiles as a new meld")
    print("  - Click 'Add to Meld' to add a single tile to a table meld")
    print("  - Click a table meld to select it as destination")
    print("  - First meld must be worth 30+ points")
    print("  - Valid melds: 3+ consecutive same-color OR same-number different colors")
    print("  - Hand is automatically sorted by color and number")
    print("  - Table melds remain in the order they were played")
    print("  - Press SPACE to draw a tile")
    print("  - Press ENTER to play selected meld")
    print("  - Press C to clear selection")
    print("  - Press S to sort hand manually")
    print("  - Press ESC to quit")
    print()
    
    game = RummikubGame(num_players=2)
    game.run()


if __name__ == "__main__":
    main()
