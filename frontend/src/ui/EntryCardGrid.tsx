import { cloneElement } from "react";
import type { ReactElement } from "react";
import { Box, Card, CardActionArea, CardContent, Stack, Typography } from "@mui/material";
import Grid from "@mui/material/GridLegacy";
import type { SvgIconProps } from "@mui/material/SvgIcon";

import { APP_TEXT_VARIANTS, ENTRY_CARD_ACTION_SX, ENTRY_CARD_ICON_SX, ENTRY_CARD_SX } from "./layout";

type BaseEntryCard = {
  title: string;
  description: string;
  accent: string;
  icon: ReactElement<SvgIconProps>;
};

type EntryCardItem =
  | (BaseEntryCard & { path: string; href?: never })
  | (BaseEntryCard & { href: string; path?: never });

type EntryCardGridProps = {
  cards: EntryCardItem[];
  onNavigate: (path: string) => void;
};

const EntryCardGrid = ({ cards, onNavigate }: EntryCardGridProps) => {
  return (
    <Grid container spacing={3} justifyContent="flex-start" alignItems="stretch">
      {cards.map((card) => (
        <Grid item xs={12} sm={6} md={6} lg={6} key={card.title} sx={{ display: "flex" }}>
          <Card elevation={2} sx={ENTRY_CARD_SX}>
            <CardActionArea
              sx={ENTRY_CARD_ACTION_SX}
              {...("href" in card
                ? {
                    component: "a",
                    href: card.href,
                    target: "_blank",
                    rel: "noopener noreferrer",
                  }
                : {
                    onClick: () => onNavigate(card.path),
                  })}
            >
              <Stack direction="row" spacing={1.5} alignItems="center" sx={{ width: "100%" }}>
                <Box sx={ENTRY_CARD_ICON_SX}>
                  {cloneElement(card.icon, { sx: { fontSize: 28, color: card.accent } })}
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant={APP_TEXT_VARIANTS.sectionTitle} sx={{ fontWeight: 600 }}>
                    {card.title}
                  </Typography>
                </Box>
              </Stack>

              <CardContent sx={{ p: 0, flex: 1, width: "100%" }}>
                <Typography variant={APP_TEXT_VARIANTS.body} color="text.secondary">
                  {card.description}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

export default EntryCardGrid;
export type { EntryCardItem };
